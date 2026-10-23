"""Crop-Stage Agronomic Growth Guidance for Rice and Maize.

Provides verified, stage-by-stage agronomic advice based on ICAR, TNAU,
IRRI, and CIMMYT recommendations:
- Irrigation and water management
- Fertilizer scheduling (NPK balancing)
- Weed management
- Proactive disease and pest monitoring

DISCLAIMER:
Growth stages and advisories are determined from user-supplied planting dates
or selected field stages. Image classification does not infer physiological growth stage.
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from datetime import date


@dataclass
class StageGuidance:
    stage_name: str
    duration_days: str
    irrigation_advice: str
    nutrient_management: str
    weed_control: str
    pest_disease_watch: str
    best_practices: List[str]


RICE_GROWTH_STAGES: List[StageGuidance] = [
    StageGuidance(
        stage_name="1. Nursery & Seedling Stage",
        duration_days="0 – 25 Days After Sowing (DAS)",
        irrigation_advice="Maintain shallow water (1–2 cm) in the nursery bed. Drain excess water during heavy downpours to prevent seedling suffocation.",
        nutrient_management="Apply well-rotted FYM (1 kg/m²) during bed preparation. Apply DAP @ 2 kg/cent at 10–12 days if seedlings show yellowing.",
        weed_control="Hand-weed at 12–15 DAS. Alternatively, apply Pretilachlor 30.7% EC (safener) @ 0.3 kg a.i./ha at 3–5 DAS.",
        pest_disease_watch="Watch for Thrips, Caseworm, and Seedling Blast. Clip top 2 cm of seedling tips before transplanting to remove stem borer egg masses.",
        best_practices=[
            "Dip seedling roots in Pseudomonas fluorescens solution (2.5 kg/ha) for 30 minutes before transplanting.",
            "Transplant seedlings at 20–25 days when they have 4–5 leaves.",
        ],
    ),
    StageGuidance(
        stage_name="2. Active Tillering & Vegetative Stage",
        duration_days="25 – 45 Days After Transplanting (DAT)",
        irrigation_advice="Practice Alternate Wetting and Drying (AWD): Allow water to drop 5 cm below soil surface before re-irrigating to 5 cm depth. Saves 25% water and promotes deep rooting.",
        nutrient_management="Apply 1st top dressing of Nitrogen (Urea @ 25% of total N) along with Zinc Sulphate (25 kg/ha) if basal application was missed.",
        weed_control="Operate mechanical cono-weeder or rotary weeder at 15 and 30 DAT. Alternatively, apply Bispyribac-sodium 10% SC @ 200 mL/ha at 15–20 DAT.",
        pest_disease_watch="Key threat window for Yellow Stem Borer ('dead hearts'), Leaf Folder, and early Bacterial Leaf Blight.",
        best_practices=[
            "Keep water depth shallow (2–3 cm) to facilitate maximum tiller emergence.",
            "Install pheromone traps @ 5/acre to monitor adult stem borer flights.",
        ],
    ),
    StageGuidance(
        stage_name="3. Panicle Initiation & Booting Stage",
        duration_days="45 – 65 Days After Transplanting (DAT)",
        irrigation_advice="CRITICAL WATER STAGE: Do not allow moisture stress. Maintain continuous shallow standing water (3–5 cm) from panicle initiation through booting.",
        nutrient_management="Apply 2nd top dressing of Nitrogen (25% of total N) combined with Potash (MOP @ 50% of total K). Potash strengthens cell walls and reduces lodging.",
        weed_control="Hand-pull any escaped grassy weeds (Echinochloa species) before they set seeds.",
        pest_disease_watch="Scout for Sheath Blight, Stem Borer, and early False Smut. Preventive spray of Copper Oxychloride 50% WP (2.5 g/L) at boot stage prevents False Smut.",
        best_practices=[
            "Strictly avoid excess urea; lush succulent growth invites sheath rot and blast.",
            "Erect bird perches (T-shaped poles, 10/acre) to encourage insectivorous birds.",
        ],
    ),
    StageGuidance(
        stage_name="4. Flowering & Heading Stage",
        duration_days="65 – 85 Days After Transplanting (DAT)",
        irrigation_advice="Ensure uninterrupted soil saturation (3–4 cm standing water). Drought stress during anthesis results in high spikelet sterility (chaffy grains).",
        nutrient_management="No nitrogen application at this stage. Foliar spray of 1% Potassium Nitrate (KNO₃) can enhance grain setting if canopy is pale.",
        weed_control="Field canopy is fully closed; manual spot rogueing only.",
        pest_disease_watch="Watch for Rice Bug (Gundhi bug) which sucks milk from developing grains, and False Smut spore balls.",
        best_practices=[
            "Avoid spraying chemical pesticides between 9:00 AM and 1:00 PM to avoid disrupting natural pollination.",
            "Install light traps along field borders to suppress nocturnal moths.",
        ],
    ),
    StageGuidance(
        stage_name="5. Grain Filling, Dough & Maturity",
        duration_days="85 – 115 Days After Transplanting (DAT)",
        irrigation_advice="Maintain thin water film during milk stage. Completely drain water 7–10 days prior to harvest to facilitate uniform grain ripening and mechanical harvesting.",
        nutrient_management="Fertilizer applications are completed. No top dressings.",
        weed_control="None needed.",
        pest_disease_watch="Watch for Brown Planthopper (BPH) 'hopper burn' in lower canopy. Part rows to inspect stem bases.",
        best_practices=[
            "Harvest when 80–85% of panicle grains have turned golden straw yellow.",
            "Thresh and dry grains promptly to 12–14% moisture content for safe storage.",
        ],
    ),
]


MAIZE_GROWTH_STAGES: List[StageGuidance] = [
    StageGuidance(
        stage_name="1. Emergence & Early Seedling (VE – V3)",
        duration_days="0 – 15 Days After Sowing (DAS)",
        irrigation_advice="Light irrigation immediately after sowing. Avoid waterlogging; maize roots are extremely sensitive to standing water (causes purple/yellow stunted seedlings).",
        nutrient_management="Apply full basal dose: 100% Phosphorus (SSP/DAP), 100% Potassium (MOP), and 1/3rd Nitrogen (Urea), plus Zinc Sulphate (25 kg/ha).",
        weed_control="Apply pre-emergence herbicide Atrazine 50% WP @ 1.0 kg a.i./ha within 2–3 days after sowing in moist soil.",
        pest_disease_watch="HIGH ALERT for Fall Armyworm (FAW) scraping on young whorls ('window-paning') and cutworms. Scout 20 plants across 5 spots.",
        best_practices=[
            "Drop dry fine sand mixed with neem cake (9:1) into central whorls to deter FAW larvae.",
            "Maintain optimal plant spacing: 60 cm between rows, 20 cm between plants.",
        ],
    ),
    StageGuidance(
        stage_name="2. Knee-High Vegetative Stage (V6 – V10)",
        duration_days="15 – 35 Days After Sowing (DAS)",
        irrigation_advice="Irrigate at 8–10 day intervals depending on soil texture. Ensure ridges and furrows facilitate rapid drainage during monsoon rains.",
        nutrient_management="Apply 2nd split of Nitrogen (1/3rd of total Urea) as side-dressing 10 cm away from plant bases followed by earthing-up.",
        weed_control="Hand-weeding or inter-cultivation with power weeder at 20–25 DAS followed by earthing up to support root stability.",
        pest_disease_watch="Watch for Fall Armyworm whorl feeding damage, Corn Leaf Aphids, and Maydis / Turcicum leaf blights on lower leaves.",
        best_practices=[
            "Earth-up soil around plant bases to prevent root lodging and suppress intra-row weeds.",
            "Apply whorl spray of Metarhizium rileyi (5 g/L) or Emamectin Benzoate 5% SG (0.4 g/L) if FAW whorl damage exceeds 10%.",
        ],
    ),
    StageGuidance(
        stage_name="3. Tasseling & Silking Stage (VT – R1)",
        duration_days="45 – 65 Days After Sowing (DAS)",
        irrigation_advice="MOST CRITICAL WATER STAGE: Water stress during silking delays silk emergence, desynchronizes pollen shed, and causes barren cobs. Irrigate without delay.",
        nutrient_management="Apply final 1/3rd Nitrogen split (Urea) at late vegetative/tasseling emergence. Avoid urea application after pollination is complete.",
        weed_control="Canopy shade suppresses most weeds; hand-rogue any tall weed escapes.",
        pest_disease_watch="Watch for FAW larvae entering ear tips, Stem Borers, and Downy Mildew / Rusts.",
        best_practices=[
            "Maintain continuous, steady soil moisture throughout the 2-week flowering window.",
            "Do not disturb plants or shake tassels during peak morning pollen shed (8:00 AM – 11:00 AM).",
        ],
    ),
    StageGuidance(
        stage_name="4. Cob Filling & Dough Stage (R2 – R4)",
        duration_days="65 – 85 Days After Sowing (DAS)",
        irrigation_advice="Maintain adequate soil moisture to ensure complete grain filling to the tip of the cob. Irrigate every 10–12 days.",
        nutrient_management="No further soil fertilizers. If foliar yellowing is observed, spray 1% Potassium Nitrate (KNO₃) to support starch translocation.",
        weed_control="None needed.",
        pest_disease_watch="Inspect cob tips for FAW or Corn Earworm feeding. Check for Curvularia leaf spot and Banded Leaf & Sheath Blight.",
        best_practices=[
            "Monitor cob development by feeling cob fullness through the husk.",
            "Scare away bird flocks during early milk and dough stages.",
        ],
    ),
    StageGuidance(
        stage_name="5. Physiological Maturity & Harvest (R6)",
        duration_days="85 – 105 Days After Sowing (DAS)",
        irrigation_advice="Withhold irrigation 10–12 days prior to harvest to promote natural drying of husks and grain.",
        nutrient_management="Fertilizer cycle completed.",
        weed_control="None needed.",
        pest_disease_watch="Watch for cob rots (Fusarium, Aspergillus) if late unseasonal rains occur.",
        best_practices=[
            "Harvest when husk leaves turn straw-yellow and dry, and a black layer forms at the grain base (kernel milk line disappears).",
            "Sun-dry de-husked cobs on clean tarpaulins until grain moisture drops below 12–13% before shelling.",
        ],
    ),
]


def estimate_current_stage(crop: str, sowing_date: date) -> Optional[StageGuidance]:
    """Estimates the current crop growth stage from planting date."""
    today = date.today()
    days_elapsed = (today - sowing_date).days
    if days_elapsed < 0:
        return None

    stages = RICE_GROWTH_STAGES if "rice" in crop.lower() else MAIZE_GROWTH_STAGES
    
    # Simple day intervals
    if "rice" in crop.lower():
        if days_elapsed <= 25:
            return stages[0]
        elif days_elapsed <= 50:
            return stages[1]
        elif days_elapsed <= 70:
            return stages[2]
        elif days_elapsed <= 90:
            return stages[3]
        else:
            return stages[4]
    else:
        if days_elapsed <= 18:
            return stages[0]
        elif days_elapsed <= 42:
            return stages[1]
        elif days_elapsed <= 65:
            return stages[2]
        elif days_elapsed <= 85:
            return stages[3]
        else:
            return stages[4]
