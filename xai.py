"""Explainable AI (XAI) module for RiceMaize AI.

Provides visual explanations of model predictions:
- Vision Transformer (DeiT-Tiny): Multi-head self-attention attribution over 16x16 patch tokens.
- Convolutional Neural Network (RiceMaizeCNN): Grad-CAM (Gradient-weighted Class Activation Mapping).
- Zero-dependency Jet colormap superimposition on PIL images.

DISCLAIMER:
Highlighted regions indicate image areas that influenced the model's prediction.
Visual explanations serve as an interpretability aid for farmers and agronomists,
and do not constitute definitive biological proof of infection.
"""

from typing import Optional, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def _jet_colormap(heatmap: np.ndarray) -> np.ndarray:
    """Zero-dependency vector Jet colormap for normalized 2D heatmap in [0, 1]."""
    h = np.clip(heatmap, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)


def blend_heatmap_with_image(
    heatmap: np.ndarray,
    original_image: Image.Image,
    alpha: float = 0.5,
) -> Image.Image:
    """Superimposes a 2D float heatmap in [0, 1] onto a PIL image."""
    orig_rgb = original_image.convert("RGB")
    w, h = orig_rgb.size

    # Resize heatmap to match original image dimensions using PIL bilinear
    heatmap_pil = Image.fromarray((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8))
    resample = getattr(Image, "Resampling", Image).BILINEAR
    heatmap_resized = heatmap_pil.resize((w, h), resample=resample)
    heatmap_norm = np.array(heatmap_resized).astype(np.float32) / 255.0

    colored_heatmap = _jet_colormap(heatmap_norm)
    orig_array = np.array(orig_rgb, dtype=np.float32)

    blended = (1.0 - alpha) * orig_array + alpha * colored_heatmap.astype(np.float32)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(blended)


def generate_vit_attention_map(
    model: torch.nn.Module,
    tensor: torch.Tensor,
) -> np.ndarray:
    """Extracts CLS-to-patch self-attention map from the final ViT transformer block.

    Args:
        model: Trained VisionTransformer instance.
        tensor: Preprocessed image tensor [1, 3, 224, 224].

    Returns:
        2D numpy array [224, 224] normalized to [0, 1].
    """
    model.eval()
    with torch.no_grad():
        _ = model(tensor)

    last_attn = getattr(model.blocks[-1].attn, "last_attn_map", None)
    if last_attn is None:
        raise RuntimeError("No cached attention map found in ViT. Ensure models.py saves last_attn_map.")

    # last_attn: [batch, num_heads, tokens, tokens] = [1, 3, 197, 197]
    # Token 0 is CLS; tokens 1..196 correspond to 14x14 spatial patches
    cls_attn = last_attn[0, :, 0, 1:]  # [3, 196]
    mean_attn = cls_attn.mean(dim=0).cpu().numpy()  # [196]

    # Reshape into 14x14 grid
    grid_size = 14
    attn_grid = mean_attn.reshape(grid_size, grid_size)

    # Normalize to [0, 1]
    attn_min, attn_max = attn_grid.min(), attn_grid.max()
    if attn_max > attn_min:
        attn_norm = (attn_grid - attn_min) / (attn_max - attn_min)
    else:
        attn_norm = np.zeros_like(attn_grid)

    return attn_norm


def generate_cnn_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """Computes Grad-CAM on the final ConvBlock of RiceMaizeCNN.

    Args:
        model: Trained RiceMaizeCNN instance.
        tensor: Preprocessed image tensor [1, 3, 224, 224].
        target_class: Target class index. If None, uses top predicted class.

    Returns:
        2D numpy array [7, 7] normalized to [0, 1].
    """
    model.eval()
    activations = []
    gradients = []

    target_layer = model.features[-1]

    def forward_hook(module, inp, out):
        activations.append(out)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    h_fwd = target_layer.register_forward_hook(forward_hook)
    if hasattr(target_layer, "register_full_backward_hook"):
        h_bwd = target_layer.register_full_backward_hook(backward_hook)
    else:
        h_bwd = target_layer.register_backward_hook(backward_hook)

    try:
        tensor_var = tensor.clone().detach().requires_grad_(True)
        logits = model(tensor_var)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1).item())

        score = logits[0, target_class]
        model.zero_grad()
        score.backward()

        act = activations[0].detach()  # [1, 256, 7, 7]
        grad = gradients[0].detach()   # [1, 256, 7, 7]

        # Global average pool of gradients -> channel weights
        weights = grad.mean(dim=(2, 3), keepdim=True)  # [1, 256, 1, 1]
        cam = F.relu((weights * act).sum(dim=1)).squeeze(0).cpu().numpy()  # [7, 7]

        c_min, c_max = cam.min(), cam.max()
        if c_max > c_min:
            cam_norm = (cam - c_min) / (c_max - c_min)
        else:
            cam_norm = np.zeros_like(cam)

        return cam_norm
    finally:
        h_fwd.remove()
        h_bwd.remove()


def explain_prediction(
    model: torch.nn.Module,
    model_type: str,
    tensor: torch.Tensor,
    original_image: Image.Image,
    target_class: Optional[int] = None,
    alpha: float = 0.45,
) -> Tuple[Optional[Image.Image], str]:
    """Generates an attribution overlay for the specified model.

    Returns:
        (blended_pil_image, caption_text) or (None, error_message)
    """
    try:
        clean_type = model_type.lower().strip()
        if clean_type in ("vit", "transformer"):
            heatmap = generate_vit_attention_map(model, tensor)
            method_name = "Self-Attention Attribution (DeiT-Tiny)"
        elif clean_type == "cnn":
            heatmap = generate_cnn_gradcam(model, tensor, target_class=target_class)
            method_name = "Grad-CAM Feature Activation (CNN)"
        elif clean_type == "hybrid":
            # For hybrid ensemble, explain using the ViT backbone (primary weight contributor)
            vit_backbone = getattr(model, "vit", model)
            heatmap = generate_vit_attention_map(vit_backbone, tensor)
            method_name = "Hybrid Attention Attribution (ViT Backbone)"
        else:
            return None, f"Unsupported model architecture for XAI: {model_type}"

        blended = blend_heatmap_with_image(heatmap, original_image, alpha=alpha)
        caption = (
            f"Visual explanation generated via {method_name}. "
            "Highlighted warm regions indicate areas of the plant image that most strongly "
            "influenced the diagnostic prediction. (Interpretability aid only; not definitive proof)."
        )
        return blended, caption

    except Exception as exc:
        return None, f"Could not generate visual explanation: {exc}"
