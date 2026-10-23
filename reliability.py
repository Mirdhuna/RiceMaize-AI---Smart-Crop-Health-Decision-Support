"""Prediction confidence calibration and reliability assessment module.

Features:
- Temperature scaling calibration (parameter T tuned on validation data only).
- Validation-based uncertainty metrics: Predictive Entropy, Top-1 Margin, Energy score.
- Out-of-Distribution (OOD) and low-confidence detection for unsupported crops / objects.
- Explicit top-3 calibrated predictions.
- Farmer-friendly cautionary notices when predictions are ambiguous or uncertain.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F

# Optimal temperature parameter tuned strictly on validation set cross-entropy (T=1.20)
VALIDATION_TEMPERATURE = 1.20

# Validation-based empirical thresholds
HIGH_CONF_THRESHOLD = 0.70
LOW_CONF_THRESHOLD = 0.45
MAX_ENTROPY_THRESHOLD = 1.80
MIN_MARGIN_THRESHOLD = 0.15


@dataclass
class ReliabilityReport:
    predicted_class: str
    top_predictions: List[Tuple[str, float]]  # [(class_name, confidence_percent)]
    calibrated_confidence: float              # in percentage (0-100)
    raw_confidence: float                     # in percentage (0-100)
    predictive_entropy: float
    confidence_margin: float                  # diff between rank 1 and rank 2
    temperature: float
    reliability_level: str                    # "High Confidence", "Moderate Confidence", "Uncertain / Expert Verification Recommended"
    is_reliable: bool
    warning_notice: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


def compute_expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    num_bins: int = 10,
) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    total_samples = len(confidences)

    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def assess_prediction_reliability(
    logits: torch.Tensor,
    class_names: List[str],
    temperature: float = VALIDATION_TEMPERATURE,
    top_k: int = 3,
) -> ReliabilityReport:
    """Calibrates logits and evaluates prediction reliability.

    Args:
        logits: Raw model output logits [1, num_classes] or [num_classes].
        class_names: List of all 17 class strings.
        temperature: Temperature scaling factor.
        top_k: Number of top candidate classes to return.

    Returns:
        ReliabilityReport with calibrated confidences and uncertainty diagnostics.
    """
    if logits.ndim == 1:
        logits = logits.unsqueeze(0)

    # 1. Raw softmax
    raw_probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    raw_conf = float(raw_probs.max()) * 100.0

    # 2. Temperature-scaled calibrated softmax
    calibrated_logits = logits / max(temperature, 1e-4)
    cal_probs = F.softmax(calibrated_logits, dim=1).squeeze(0).cpu().numpy()

    sorted_indices = np.argsort(cal_probs)[::-1]
    top_indices = sorted_indices[:top_k]

    pred_idx = top_indices[0]
    pred_class = class_names[pred_idx]
    cal_conf = float(cal_probs[pred_idx]) * 100.0

    # 3. Uncertainty metrics
    # Shannon Entropy (nats): H = -sum(p * log(p))
    epsilon = 1e-8
    entropy = float(-np.sum(cal_probs * np.log(cal_probs + epsilon)))

    # Margin between rank-1 and rank-2
    margin = float(cal_probs[sorted_indices[0]] - cal_probs[sorted_indices[1]]) if len(sorted_indices) > 1 else 1.0

    # Energy score: E(x) = -T * log(sum(exp(logits / T)))
    with torch.no_grad():
        energy = float(-temperature * torch.logsumexp(logits / temperature, dim=1).item())

    # Build top-k predictions
    top_preds = [
        (class_names[idx], round(float(cal_probs[idx]) * 100.0, 2))
        for idx in top_indices
    ]

    # 4. Reliability categorization & OOD warning generation
    warning_notice = None
    cal_fraction = cal_probs[pred_idx]

    if cal_fraction < LOW_CONF_THRESHOLD or margin < MIN_MARGIN_THRESHOLD or entropy > MAX_ENTROPY_THRESHOLD:
        reliability_level = "Uncertain / Expert Verification Recommended"
        is_reliable = False
        warning_notice = (
            "⚠️ **Low Diagnostic Confidence**: The model is uncertain about this classification. "
            f"The top candidate has only {cal_conf:.1f}% confidence with high prediction entropy ({entropy:.2f}). "
            "This may occur if the photograph contains an unsupported crop, unfamiliar disease stage, "
            "or non-agricultural objects. Do not apply intensive chemical treatments without consulting "
            "a local agricultural extension officer."
        )
    elif cal_fraction < HIGH_CONF_THRESHOLD or margin < 0.30:
        reliability_level = "Moderate Confidence"
        is_reliable = True
        runner_up = class_names[sorted_indices[1]]
        runner_up_pct = cal_probs[sorted_indices[1]] * 100.0
        warning_notice = (
            f"ℹ️ **Moderate Confidence**: Primary match is {pred_class.split('/')[-1]} ({cal_conf:.1f}%), "
            f"with a close secondary possibility of {runner_up.split('/')[-1]} ({runner_up_pct:.1f}%). "
            "Carefully cross-reference the leaf symptoms in the Field Guide tab."
        )
    else:
        reliability_level = "High Confidence"
        is_reliable = True

    metrics = {
        "calibrated_confidence_pct": round(cal_conf, 2),
        "raw_confidence_pct": round(raw_conf, 2),
        "predictive_entropy": round(entropy, 3),
        "confidence_margin": round(margin, 3),
        "free_energy_score": round(energy, 2),
        "temperature_applied": temperature,
    }

    return ReliabilityReport(
        predicted_class=pred_class,
        top_predictions=top_preds,
        calibrated_confidence=cal_conf,
        raw_confidence=raw_conf,
        predictive_entropy=entropy,
        confidence_margin=margin,
        temperature=temperature,
        reliability_level=reliability_level,
        is_reliable=is_reliable,
        warning_notice=warning_notice,
        metrics=metrics,
    )
