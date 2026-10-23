"""Weather-Aware Epidemiological Risk Advisory Module.

Translates current microclimate conditions (Temperature, Humidity, Rainfall, Cloud Cover)
into prospective disease and insect-pest environmental risk indicators based on verified
ICAR, TNAU, and IRRI epidemiology forecasting models.

DISCLAIMER:
Weather-aware alerts represent prospective environmental risk indicators based on microclimatic
favorability for pathogen sporulation or pest breeding. They do not represent confirmed disease diagnoses.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class PathogenRiskItem:
    condition_name: str
    risk_level: str  # "Low", "Moderate", "High", "Critical"
    favorable_factors: str
    preventive_recommendation: str


@dataclass
class WeatherRiskReport:
    crop: str
    overall_environmental_risk: str
    temperature_c: float
    humidity_pct: float
    rainfall_mm: float
    cloud_cover: str
    high_risk_threats: List[PathogenRiskItem] = field(default_factory=list)
    moderate_risk_threats: List[PathogenRiskItem] = field(default_factory=list)
    general_agronomic_advice: List[str] = field(default_factory=list)
    disclaimer: str = (
        "Environmental risk advisories indicate prospective microclimate favorability for pathogen sporulation "
        "and pest multiplication. They are early-warning risk indicators, not confirmed diagnoses."
    )


def evaluate_weather_disease_risks(
    crop: str,
    temperature_c: float,
    humidity_pct: float,
    rainfall_mm: float,
    cloud_cover: str = "Partly Cloudy",
) -> WeatherRiskReport:
    """Evaluates prospective disease and pest threats from weather parameters."""
    is_rice = "rice" in crop.lower()
    high_threats = []
    mod_threats = []
    advice = []

    # 1. High Humidity + Warm Conditions (>80% RH, 24-32°C)
    if humidity_pct >= 85.0:
        if is_rice:
            if 24.0 <= temperature_c <= 32.0:
                high_threats.append(PathogenRiskItem(
                    condition_name="Bacterial Leaf Blight (BLB)",
                    risk_level="High",
                    favorable_factors=f"Sustained high humidity ({humidity_pct:.0f}%) and warm temperatures ({temperature_c:.1f}°C) accelerate Xanthomonas multiplication.",
                    preventive_recommendation="Avoid excess urea top-dressing. Maintain shallow field water (2–3 cm) and ensure proper bund drainage.",
                ))
            if temperature_c >= 26.0:
                high_threats.append(PathogenRiskItem(
                    condition_name="Sheath Blight",
                    risk_level="High",
                    favorable_factors=f"High relative humidity ({humidity_pct:.0f}%) creates a humid microclimate in dense lower leaf canopies.",
                    preventive_recommendation="Avoid dense planting. Apply Trichoderma-enriched manure and spray Pseudomonas fluorescens targeting stem bases.",
                ))
            if "Overcast" in cloud_cover or rainfall_mm > 15.0:
                high_threats.append(PathogenRiskItem(
                    condition_name="False Smut",
                    risk_level="High",
                    favorable_factors=f"Overcast skies and high moisture during reproductive stages trigger airborne chlamydospore dispersal.",
                    preventive_recommendation="If crop is at booting stage, apply preventive spray of Copper Oxychloride 50% WP (2.5 g/L).",
                ))
        else:  # Maize
            if 20.0 <= temperature_c <= 30.0:
                high_threats.append(PathogenRiskItem(
                    condition_name="Maydis & Turcicum Leaf Blights",
                    risk_level="High",
                    favorable_factors=f"Warm, humid conditions ({humidity_pct:.0f}% RH, {temperature_c:.1f}°C) favor Exserohilum and Bipolaris conidia germination.",
                    preventive_recommendation="Ensure unobstructed field drainage. Avoid late evening sprinkler irrigation that prolongs leaf wetness.",
                ))
            if rainfall_mm > 25.0:
                mod_threats.append(PathogenRiskItem(
                    condition_name="Sorghum Downy Mildew",
                    risk_level="Moderate",
                    favorable_factors=f"Water accumulation and high moisture foster oospore germination in soil.",
                    preventive_recommendation="Ensure active field drainage furrows to prevent water stagnation around young whorls.",
                ))

    # 2. Warm & Dry Spells (<65% RH, >28°C)
    elif humidity_pct < 65.0 and temperature_c >= 28.0:
        if is_rice:
            mod_threats.append(PathogenRiskItem(
                condition_name="Rice Thrips & Leaf Skipper",
                risk_level="Moderate",
                favorable_factors=f"Dry warm spells ({temperature_c:.1f}°C, {humidity_pct:.0f}% RH) stimulate thrips reproduction in young seedlings.",
                preventive_recommendation="Submerge nursery beds briefly or spray neem oil (3 mL/L) to prevent leaf curling.",
            ))
        else:
            high_threats.append(PathogenRiskItem(
                condition_name="Corn Leaf Aphid",
                risk_level="High",
                favorable_factors=f"Warm, dry weather ({temperature_c:.1f}°C) accelerates aphid colony buildup inside central whorls.",
                preventive_recommendation="Conserve ladybird beetles and lacewings. Spray 5% Neem Seed Kernel Extract (NSKE) if aphid colonies appear.",
            ))
            mod_threats.append(PathogenRiskItem(
                condition_name="Fall Armyworm (FAW)",
                risk_level="Moderate",
                favorable_factors=f"Temperatures above 28°C accelerate FAW larval feeding cycles.",
                preventive_recommendation="Inspect 20 plants weekly across the field. Place neem cake + sand in whorls.",
            ))

    # General agronomic advice based on rainfall
    if rainfall_mm > 40.0:
        advice.append("Heavy recent rainfall detected: Check field drainage channels immediately to prevent waterlogging.")
        advice.append("Postpone foliar pesticide sprays until foliage is dry to avoid chemical wash-off.")
    elif rainfall_mm < 5.0 and humidity_pct < 55.0:
        advice.append("Dry conditions: Ensure critical irrigation during reproductive stages (panicle initiation for rice, silking for maize).")

    # Overall risk determination
    if len(high_threats) >= 2:
        overall_risk = "Elevated Environmental Risk"
    elif len(high_threats) == 1 or len(mod_threats) >= 2:
        overall_risk = "Moderate Environmental Risk"
    else:
        overall_risk = "Low Environmental Risk"

    return WeatherRiskReport(
        crop=crop,
        overall_environmental_risk=overall_risk,
        temperature_c=temperature_c,
        humidity_pct=humidity_pct,
        rainfall_mm=rainfall_mm,
        cloud_cover=cloud_cover,
        high_risk_threats=high_threats,
        moderate_risk_threats=mod_threats,
        general_agronomic_advice=advice,
    )
