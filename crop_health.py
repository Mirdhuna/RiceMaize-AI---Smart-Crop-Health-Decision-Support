"""Crop Health Assessment Module for RiceMaize AI.

Combines the visual diagnostic outcome with optional farmer field observations
(crop stage, canopy spread, recent weather conditions, and leaf vigor symptoms)
to provide a holistic, agronomic field health evaluation.

DISCLAIMER:
This assessment provides preliminary decision support based on photographic
classification and user-supplied field conditions. It does not measure exact
physiological severity or replace laboratory soil/tissue diagnostics.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class HealthAssessmentResult:
    health_status: str        # "Optimal Health / Vigorous", "Moderate Stress / Action Advised", "Severe Agronomic Risk", "Assessment Uncertain"
    risk_level: str          # "Low", "Medium", "High", "Critical"
    summary: str
    contributing_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    limitations_disclaimer: str = (
        "Preliminary health assessment based on photographic recognition and farmer-reported "
        "field observations. Exact disease severity percentages and physiological metrics require "
        "on-site field sampling and laboratory confirmation."
    )


def assess_crop_health(
    predicted_class: str,
    confidence: float,
    crop_stage: str = "Not Specified",
    spread_level: str = "Isolated / Few Plants (<5%)",
    weather_condition: str = "Moderate / Normal",
    leaf_symptoms: Optional[List[str]] = None,
) -> HealthAssessmentResult:
    """Evaluates crop health status by synthesizing visual classification and agronomic context."""
    leaf_symptoms = leaf_symptoms or []
    factors = []
    actions = []

    is_healthy = "Healthy" in predicted_class
    is_pest = "Insect-pests" in predicted_class
    is_disease = "Disease" in predicted_class

    # Confidence check
    if confidence < 50.0:
        return HealthAssessmentResult(
            health_status="Assessment Uncertain",
            risk_level="Medium",
            summary="Diagnostic confidence is too low to establish a definitive health assessment.",
            contributing_factors=["Model confidence below threshold (ambiguous or low-resolution image)."],
            recommended_actions=[
                "Upload a clearer, close-up photograph of the leaf symptoms.",
                "Consult local village agricultural extension officer before taking chemical measures.",
            ],
        )

    # 1. Base classification factors
    if is_healthy:
        health_status = "Optimal Health / Vigorous"
        risk_level = "Low"
        summary = "Crop displays healthy foliage and normal photosynthetic appearance."
        factors.append("No active lesions or pest defoliation detected on the submitted sample.")
        actions.append("Maintain regular nutrient scheduling and weekly field scouting.")
    elif is_pest:
        health_status = "Pest Infestation Detected"
        risk_level = "High" if "Critical" in spread_level or "Widespread" in spread_level else "Medium"
        summary = f"Active pest threat identified ({predicted_class.split('/')[-1]})."
        factors.append(f"Visual presence of insect pest or characteristic feeding damage.")
        actions.append("Implement Integrated Pest Management (IPM) starting with pheromone/light traps and biologicals.")
    else:
        health_status = "Fungal/Bacterial Infection Detected"
        risk_level = "High" if "Widespread" in spread_level else "Medium"
        summary = f"Foliar pathogen activity identified ({predicted_class.split('/')[-1]})."
        factors.append("Presence of leaf lesions, spots, or bacterial water-soaked streaks.")
        actions.append("Target infected plots with recommended biocontrol or label-compliant chemical fungicides.")

    # 2. Field Spread modifier
    if "Widespread" in spread_level or ">20%" in spread_level:
        factors.append(f"Field spread reported as widespread ({spread_level}), elevating crop vulnerability.")
        if risk_level != "Critical":
            risk_level = "High"
        actions.append("Prioritize block-level treatment rather than spot spraying.")
    elif "Patchy" in spread_level:
        factors.append(f"Infection is localized in field patches ({spread_level}).")
        actions.append("Isolate affected patches to prevent pathogen movement into adjacent healthy rows.")

    # 3. Weather context modifier
    if "Humid" in weather_condition or "Rainy" in weather_condition:
        factors.append(f"High humidity or rainy weather ({weather_condition}) promotes rapid spore germination and bacterial ooze.")
        if not is_healthy:
            actions.append("Temporarily avoid nitrogen top-dressing; damp canopies accelerate blight expansion.")
    elif "Hot & Dry" in weather_condition:
        factors.append(f"Warm, dry conditions ({weather_condition}) favor aphid reproduction and spider mite build-up.")

    # 4. Crop stage vulnerability modifier
    if crop_stage in ("Flowering / Heading", "Tasseling & Silking", "Grain Filling"):
        factors.append(f"Crop is at critical reproductive stage ({crop_stage}) where foliage loss directly impacts grain yield.")
        if not is_healthy:
            actions.append("Intervene promptly to preserve upper canopy and flag leaves responsible for grain filling.")

    return HealthAssessmentResult(
        health_status=health_status,
        risk_level=risk_level,
        summary=summary,
        contributing_factors=factors,
        recommended_actions=actions,
    )
