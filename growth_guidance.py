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


def estimate_current_stage(crop: str, sowing_date: date, lang: str = "en") -> Optional[StageGuidance]:
    """Estimates the current crop growth stage from planting date."""
    today = date.today()
    days_elapsed = (today - sowing_date).days
    if days_elapsed < 0:
        return None

    stages = get_growth_stages_for_crop(crop, lang=lang)
    
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


RICE_GROWTH_STAGES_TA: List[StageGuidance] = [
    StageGuidance(
        stage_name="1. நாற்றங்கால் மற்றும் முளைக்கும் பருவம்",
        duration_days="0 – 25 நாட்கள் (விதைத்ததிலிருந்து)",
        irrigation_advice="நாற்றங்காலில் மெல்லிய அளவில் (1–2 செ.மீ) நீர் தேங்குமாறு பார்த்துக் கொள்ளவும். அதிக கனமழை பெய்தால் நாற்று அழுகாமல் இருக்க உபரி நீரை உடனடியாக வடித்துவிடவும்.",
        nutrient_management="நாற்றங்கால் தயார் செய்யும்போது சென்ட்டுக்கு 10 கிலோ மக்கிய தொழு உரம் இடவும். நாற்றுகள் மஞ்சள் நிறமாக மாறினால் 10–12 நாட்களில் சென்ட்டுக்கு 2 கிலோ டி.ஏ.பி (DAP) இடவும்.",
        weed_control="12–15 நாட்களில் கையால் களை எடுக்கவும். அல்லது விதைத்த 3–5 நாட்களில் பிரிட்டிலாக்ளோர் (Pretilachlor) களைக்கொல்லியைப் பயன்படுத்தவும்.",
        pest_disease_watch="இலைப்பேன், கூண்டுப்புழு மற்றும் குலைநோய் தாக்குதலைக் கண்காணிக்கவும். தண்டு துளைப்பான் முட்டைக்குவியல்களை அழிக்க நாற்று நடும் முன் நுனியை 2 செ.மீ கிள்ளி நடவும்.",
        best_practices=[
            "நாற்று நடும் முன் வேர்களை சூடோமோனாஸ் கரைசலில் (ஹெக்டேருக்கு 2.5 கிலோ) 30 நிமிடங்கள் நனைத்து நடவும்.",
            "4–5 இலைகள் வந்தவுடன் (20–25 நாட்களில்) சரியான இடைவெளியில் நடவு செய்யவும்.",
        ],
    ),
    StageGuidance(
        stage_name="2. தூர்கட்டும் பருவம் (வளர்ச்சி நிலை)",
        duration_days="25 – 45 நாட்கள் (நட்டதிலிருந்து)",
        irrigation_advice="காய்ச்சலும் பாய்ச்சலுமாக நீர் பாய்ச்சவும்: மண்ணின் மேற்பரப்பில் நீர் வடிந்து லேசான கீறல் விழும்போது மீண்டும் 5 செ.மீ அளவுக்கு நீர் பாய்ச்சவும். இது 25% தண்ணீரை சேமித்து வேர்களை ஆழமாக வளர்க்கும்.",
        nutrient_management="முதலாம் மேலுரமாக பரிந்துரைக்கப்பட்ட தழைச்சத்தில் 25% யூரியா மற்றும் துத்தநாக சல்பேட் (ஹெக்டேருக்கு 25 கிலோ) இடவும்.",
        weed_control="நட்ட 15 மற்றும் 30-ம் நாட்களில் கோனோவீடர் (உருளைக் களைக்கருவி) மூலம் களைகளை மண்ணுக்குள் அமுக்கிவிடவும்.",
        pest_disease_watch="மஞ்சள் தண்டு துளைப்பான் ('குருத்து கருகல்' / 'Dead heart'), இலைச்சுருட்டுப் புழு மற்றும் ஆரம்ப கால இலைக்கருகல் நோயைக் கண்காணிக்கவும்.",
        best_practices=[
            "அதிக தூர்கள் வெடிக்க வயலில் குறைந்த அளவு (2–3 செ.மீ) நீர் நிறுத்துவது நல்லது.",
            "தண்டு துளைப்பான் அந்துப்பூச்சிகளைக் கண்காணிக்க ஏக்கருக்கு 5 இனக்கவர்ச்சிப் பொறிகளை வைக்கவும்.",
        ],
    ),
    StageGuidance(
        stage_name="3. கதிர் உருவாகும் மற்றும் சூல் பருவம்",
        duration_days="45 – 65 நாட்கள் (நட்டதிலிருந்து)",
        irrigation_advice="மிக முக்கியமான நீர்த் தேவைப் பருவம்: பயிரில் தண்ணீர் பற்றாக்குறை ஏற்படவே கூடாது. கதிர் உருவாகும் காலம் முதல் பூக்கும் வரை தொடர்ந்து 3–5 செ.மீ நீர் இருக்குமாறு பராமரிக்கவும்.",
        nutrient_management="இரண்டாம் மேலுரமாக எஞ்சிய யூரியாவுடன் 50% பொட்டாஷ் உரம் (MOP) சேர்த்து இடவும். பொட்டாஷ் இடுவதால் தண்டு வலிமை அடைந்து பயிர் சாயாமல் காக்கும்.",
        weed_control="கதிர் வெளிவரும் முன் தப்பிய பெரிய களைகளைக் கையால் பிடுங்கி அப்புறப்படுத்தவும்.",
        pest_disease_watch="இலை உறை அழுகல் நோய், தண்டு துளைப்பான் மற்றும் போலி கரிப்பூட்டை நோய் வர வாய்ப்புள்ளது. சூல் பருவத்தில் காப்பர் ஆக்ஸிகுளோரைடு தெளிப்பது கரிப்பூட்டை நோயைத் தடுக்கும்.",
        best_practices=[
            "அதிகப்படியான யூரியாவைத் தவிர்க்கவும்; அதிக தழைச்சத்து பூச்சி மற்றும் நோய்களை ஈர்க்கும்.",
            "வயலில் ஏக்கருக்கு 10 இடங்களில் பறவைத் தாங்கிகள் ('T' வடிவ குச்சிகள்) அமைத்து பூச்சிகளை உண்ணும் பறவைகளை ஈர்க்கவும்.",
        ],
    ),
    StageGuidance(
        stage_name="4. பூக்கும் மற்றும் பால் பிடிக்கும் பருவம்",
        duration_days="65 – 85 நாட்கள் (நட்டதிலிருந்து)",
        irrigation_advice="வயலில் நீர் வற்றிவிடக் கூடாது (3–4 செ.மீ நீர்). பூக்கும் தருணத்தில் வறட்சி ஏற்பட்டால் நெல் மணிகள் பதராக மாறிவிடும்.",
        nutrient_management="இப்பருவத்தில் யூரியா இடக்கூடாது. பயிர் வெளிறி இருந்தால் 1% பொட்டாசியம் நைட்ரேட் (KNO₃) இலைவழித் தெளிப்பாகத் தெளிக்கலாம்.",
        weed_control="பயிர் நன்கு மூடியிருப்பதால் களை எடுக்கத் தேவையில்லை.",
        pest_disease_watch="பால்கட்டும் பருவத்தில் நெல் மணிகளை உறிஞ்சும் 'நெல் நாவாய்ப்பூச்சி' (கந்தகப் பூச்சி) தாக்குதலைக் கண்காணிக்கவும்.",
        best_practices=[
            "மகரந்தச் சேர்க்கை பாதிக்கப்படாமல் இருக்க காலை 9:00 மணி முதல் மதியம் 1:00 மணி வரை மருந்து தெளிப்பதைத் தவிர்க்கவும்.",
            "இரவு நேர அந்துப்பூச்சிகளைக் கவர்ந்து அழிக்க வயல் ஓரங்களில் விளக்குப் பொறிகளை வைக்கவும்.",
        ],
    ),
    StageGuidance(
        stage_name="5. தானியம் முதிரும் பருவம் & அறுவடை நிலை",
        duration_days="85 – 115 நாட்கள் (நட்டதிலிருந்து)",
        irrigation_advice="பால் இறுகியவுடன் மெல்லிய நீர் மட்டும் பாய்ச்சவும். அறுவடைக்கு 7 முதல் 10 நாட்களுக்கு முன்பாக வயலில் உள்ள தண்ணீரை முழுமையாக வடித்துவிடவும்.",
        nutrient_management="உரப் பயன்பாடு முடிந்துவிட்டது.",
        weed_control="தேவையில்லை.",
        pest_disease_watch="பயிரின் அடிப்பகுதியில் புகையான் (Brown Planthopper) பூச்சி தாக்குதல் உள்ளதா என பயிரைப் பிரித்துப் பார்க்கவும்.",
        best_practices=[
            "கதிரில் உள்ள 80–85% நெல்மணிகள் வைக்கோல் மஞ்சள் நிறமாக மாறியவுடன் அறுவடை செய்யவும்.",
            "அறுவடை செய்த நெல்லை 12–14% ஈரப்பதம் வரும் வரை நன்கு உலர்த்திப் பின் சேமிக்கவும்.",
        ],
    ),
]


MAIZE_GROWTH_STAGES_TA: List[StageGuidance] = [
    StageGuidance(
        stage_name="1. முளைத்தல் மற்றும் ஆரம்ப பயிர் நிலை (0 – 15 நாட்கள்)",
        duration_days="0 – 15 நாட்கள் (விதைத்ததிலிருந்து)",
        irrigation_advice="விதைத்தவுடன் ஒரு உயிர்த் தண்ணீரும், 3-ம் நாள் முளைப்புத் தண்ணீரும் பாய்ச்சவும். மக்காச்சோளப் பயிர் தண்ணீர் தேங்குவதைத் தாங்காது; நீர் தேங்கினால் பயிர் மஞ்சள் நிறமாகி அழுகிவிடும்.",
        nutrient_management="முழு மணிச்சத்து (DAP/SSP), முழு சாம்பல் சத்து (பொட்டாஷ்) மற்றும் 1/3 பங்கு தழைச்சத்து (யூரியா) ஆகியவற்றை அடியுரமாக இடவும். ஏக்கருக்கு 10 கிலோ துத்தநாக சல்பேட் இடவும்.",
        weed_control="விதைத்த 2–3 நாட்களுக்குள் மண்ணில் ஈரம் இருக்கும்போது அட்ராசின் (Atrazine 50% WP) ஏக்கருக்கு 400 கிராம் தெளிக்கவும்.",
        pest_disease_watch="அதி தீவிர எச்சரிக்கை: படைப்புழு (Fall Armyworm) தாக்குதல் இளம் குருத்துகளில் தொடங்கும். குருத்தில் சிறிய ஓட்டைகள் மற்றும் மரத்தூள் போன்ற கழிவுகள் உள்ளதா என கவனிக்கவும்.",
        best_practices=[
            "படைப்புழுவை விரட்ட வேப்பம்பிண்ணாக்கு மற்றும் மணலை (1:9 விகிதத்தில்) கலந்து இளம் குருத்துகளின் நடுவே போடவும்.",
            "செடிக்குச் செடி 20 செ.மீ, வரிசைக்கு வரிசை 60 செ.மீ இடைவெளி பராமரிக்கவும்.",
        ],
    ),
    StageGuidance(
        stage_name="2. முழங்கால் உயர வளர்ச்சிப் பருவம் (15 – 35 நாட்கள்)",
        duration_days="15 – 35 நாட்கள் (விதைத்ததிலிருந்து)",
        irrigation_advice="மண்ணின் தன்மைக்கேற்ப 8–10 நாட்களுக்கு ஒருமுறை பாசனம் செய்யவும். மழைக்காலத்தில் பாத்திகளில் நீர் தேங்காமல் வெளியேற வாய்க்கால் அமைக்கவும்.",
        nutrient_management="விதைத்த 25–30 நாட்களில் இரண்டாம் தவணை யூரியாவை (1/3 பங்கு) செடியின் தண்டுக்கு அருகில் வைத்து மண் அணைக்கவும்.",
        weed_control="20–25 நாட்களில் கையால் களை எடுத்து அல்லது விசைக்கருவி கொண்டு களை நீக்கி செடிகளுக்கு மண் அணைக்கவும்.",
        pest_disease_watch="படைப்புழு குருத்துத் தாக்குதல், அசுவினி மற்றும் அடி இலைகளில் இலைக்கருகல் நோய் உள்ளதா என கண்காணிக்கவும்.",
        best_practices=[
            "வேர்கள் பலப்படவும் பயிர் சாயாமல் இருக்கவும் செடிகளைச் சுற்றி மண் அணைப்பது கட்டாயம்.",
            "படைப்புழு தாக்கம் இருந்தால் இமாமெக்டின் பென்சோயேட் (Emamectin Benzoate 5% SG) 10 லிட்டர் தண்ணீருக்கு 4 கிராம் வீதம் குருத்தில் படுமாறு தெளிக்கவும்.",
        ],
    ),
    StageGuidance(
        stage_name="3. ஆண் பூ (கதிர்) மற்றும் பெண் பூ (சில்க்) பருவம்",
        duration_days="45 – 65 நாட்கள் (விதைத்ததிலிருந்து)",
        irrigation_advice="மிக முக்கியமான நீர்த் தேவைப் பருவம்: பூக்கும் போது தண்ணீர் பற்றாக்குறை ஏற்பட்டால் கதிர்களில் மணிகள் பிடிக்காமல் பதராகிவிடும். தடையின்றி பாசனம் செய்யவும்.",
        nutrient_management="கடைசி 1/3 பங்கு யூரியாவை ஆண் பூ வெளிவரும் முன் இடவும். பூக்கள் மலர்ந்த பின் யூரியா இடக்கூடாது.",
        weed_control="பயிர் நன்கு மூடியிருக்கும்; தப்பிய பெரிய களைகளை மட்டும் அகற்றவும்.",
        pest_disease_watch="படைப்புழு கதிர் முனைகளில் நுழைகிறதா என பார்க்கவும். சாம்பல் பூஞ்சாண நோய் மற்றும் துரு நோயைக் கண்காணிக்கவும்.",
        best_practices=[
            "பூக்கும் 2 வார காலத்திற்கு மண்ணில் போதிய ஈரப்பதத்தை தொடர்ந்து பராமரிக்கவும்.",
            "காலை 8:00 முதல் 11:00 மணி வரை மகரந்தச் சேர்க்கை நடைபெறுவதால் செடிகளை அசைக்கவோ மருந்து தெளிக்கவோ கூடாது.",
        ],
    ),
    StageGuidance(
        stage_name="4. கதிர் பால் பிடிக்கும் பருவம் (65 – 85 நாட்கள்)",
        duration_days="65 – 85 நாட்கள் (விதைத்ததிலிருந்து)",
        irrigation_advice="கதிரின் நுனி வரை மணிகள் முழுமையாக நிரம்புவதற்கு 10 நாட்களுக்கு ஒருமுறை மிதமான பாசனம் தேவை.",
        nutrient_management="மண்ணில் உரம் இடத் தேவையில்லை. இலைகள் மஞ்சள் நிறமானால் 1% பொட்டாசியம் நைட்ரேட் தெளிக்கலாம்.",
        weed_control="களை கட்டுப்பாடு தேவையில்லை.",
        pest_disease_watch="கதிர் முனைகளில் புழுக்கள் மற்றும் கதிர் அழுகல் நோய் உள்ளதா என சோதிக்கவும்.",
        best_practices=[
            "கதிர்களைத் தொட்டுப் பார்த்து மணிகள் நன்றாக பிடித்துள்ளதா என உறுதி செய்யவும்.",
            "பால் பருவத்தில் கதிர்களைக் கொத்தும் கிளிகள் மற்றும் பறவைகளை விரட்ட ஒலி எழுப்பும் கருவிகளைப் பயன்படுத்தவும்.",
        ],
    ),
    StageGuidance(
        stage_name="5. கதிர் முதிர்ச்சி மற்றும் அறுவடை நிலை (85 – 105 நாட்கள்)",
        duration_days="85 – 105 நாட்கள் (விதைத்ததிலிருந்து)",
        irrigation_advice="அறுவடைக்கு 10–12 நாட்களுக்கு முன்பாக பாசனத்தை நிறுத்தி கதிர்கள் காய்வதற்கு அனுமதிக்கவும்.",
        nutrient_management="உரமிடுதல் தேவையில்லை.",
        weed_control="தேவையில்லை.",
        pest_disease_watch="மழைக் காலங்களில் கதிர்களில் பூஞ்சாணம் பிடிக்காமல் பாதுகாக்க உடனே அறுவடை செய்யவும்.",
        best_practices=[
            "கதிரின் மேல் தோகை காய்ந்து வைக்கோல் நிறமாக மாறும் போதும், விதையின் அடிப்பகுதியில் கருப்பு வளையம் தோன்றும் போதும் அறுவடை செய்யவும்.",
            "கதிர்களை உறித்து தார்ப்பாயில் நன்கு வெயிலில் காயவைத்து 12% ஈரப்பதத்திற்குள் கொண்டுவந்து மணிகளைப் பிரிக்கவும்.",
        ],
    ),
]


def get_growth_stages_for_crop(crop: str, lang: str = "en") -> List[StageGuidance]:
    """Returns stage guidance in English or Tamil."""
    is_rice = "rice" in crop.lower()
    if lang == "ta":
        return RICE_GROWTH_STAGES_TA if is_rice else MAIZE_GROWTH_STAGES_TA
    return RICE_GROWTH_STAGES if is_rice else MAIZE_GROWTH_STAGES
