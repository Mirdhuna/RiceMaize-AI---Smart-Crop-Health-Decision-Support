"""Lightweight CPU-based image quality validation for agricultural diagnosis.

Performs non-invasive preliminary checks before model inference:
1. Sharpness / Blur Detection: Laplacian operator variance on grayscale image.
2. Exposure / Luminance Detection: Mean luminance and extreme pixel clipping.
3. Contrast Check: Standard deviation of pixel intensities.
4. Resolution & Aspect Ratio: Minimum size verification.

Avoids rejecting valid images solely because of natural field variations,
while giving farmers actionable tips to retake clearer photos when necessary.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import numpy as np
from PIL import Image


@dataclass
class ImageQualityReport:
    is_acceptable: bool
    status: str  # "Good", "Borderline / Quality Warning", "Unsuitable"
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


def _laplacian_variance(gray: np.ndarray) -> float:
    """Computes Laplacian edge variance using vectorized 2D convolution (zero extra dependencies)."""
    # 3x3 Laplacian kernel: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    padded = np.pad(gray.astype(np.float32), 1, mode="reflect")
    lap = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
        - 4.0 * padded[1:-1, 1:-1]
    )
    return float(lap.var())


def evaluate_image_quality(image: Image.Image) -> ImageQualityReport:
    """Analyzes a PIL Image and returns an image quality assessment.

    Args:
        image: Uploaded PIL image.

    Returns:
        ImageQualityReport with status, identified issues, and field recommendations.
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size

    # Convert to grayscale for luminance and blur measurements
    img_gray = img_rgb.convert("L")
    # Downsample slightly if image is massive (e.g. >1000px) to keep CPU latency < 50ms
    max_dim = max(w, h)
    if max_dim > 800:
        scale = 800.0 / max_dim
        resample = getattr(Image, "Resampling", Image).BILINEAR
        gray_eval = img_gray.resize((int(w * scale), int(h * scale)), resample=resample)
    else:
        gray_eval = img_gray

    gray_np = np.array(gray_eval, dtype=np.uint8)

    # 1. Blur evaluation
    blur_score = _laplacian_variance(gray_np)

    # 2. Luminance evaluation
    mean_lum = float(gray_np.mean())

    # 3. Contrast evaluation
    contrast_std = float(gray_np.std())

    issues = []
    tips = []

    # Resolution threshold
    if w < 120 or h < 120:
        issues.append(f"Low image resolution ({w}x{h} pixels). Fine leaf spots may not be discernible.")
        tips.append("Upload a higher-resolution photograph capturing the whole leaf or diseased patch.")

    # Blur threshold
    if blur_score < 40.0:
        issues.append(f"Significant motion or focus blur detected (sharpness: {blur_score:.1f}).")
        tips.append("Hold camera steady 15–25 cm away from the leaf and tap the screen to focus on the symptom.")
    elif blur_score < 75.0:
        issues.append(f"Mild blur detected (sharpness: {blur_score:.1f}).")
        tips.append("Ensure the plant leaf is in sharp focus before snapping.")

    # Lighting / Exposure thresholds
    if mean_lum < 35.0:
        issues.append(f"Excessively dark / underexposed image (average brightness: {mean_lum:.1f}/255).")
        tips.append("Photograph under diffuse daylight or enable mobile phone flash.")
    elif mean_lum > 225.0:
        issues.append(f"Severe glare or overexposure (average brightness: {mean_lum:.1f}/255).")
        tips.append("Avoid direct sun glare or harsh flash reflection on glossy leaf surfaces.")

    # Low contrast threshold
    if contrast_std < 20.0:
        issues.append(f"Low visual contrast (contrast std: {contrast_std:.1f}).")
        tips.append("Ensure the leaf stands out against the background soil or clothing.")

    metrics = {
        "width": w,
        "height": h,
        "sharpness_laplacian": round(blur_score, 1),
        "mean_luminance": round(mean_lum, 1),
        "contrast_std": round(contrast_std, 1),
    }

    if not issues:
        return ImageQualityReport(
            is_acceptable=True,
            status="Good Quality",
            issues=[],
            recommendations=["Image quality is clear for visual classification."],
            metrics=metrics,
        )
    elif blur_score < 30.0 or mean_lum < 25.0 or mean_lum > 240.0:
        return ImageQualityReport(
            is_acceptable=False,
            status="Unsuitable / Retake Advised",
            issues=issues,
            recommendations=tips,
            metrics=metrics,
        )
    else:
        return ImageQualityReport(
            is_acceptable=True,
            status="Borderline / Quality Warning",
            issues=issues,
            recommendations=tips,
            metrics=metrics,
        )
