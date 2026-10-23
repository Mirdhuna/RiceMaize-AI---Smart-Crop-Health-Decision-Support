import sys
from pathlib import Path
from datetime import date

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import json
import streamlit as st
from PIL import Image
import torch

from dataset import load_class_names
from models import build_model
from torch_dataset import preprocess_pil
from xai import explain_prediction
from image_quality import evaluate_image_quality
from reliability import assess_prediction_reliability, VALIDATION_TEMPERATURE
from crop_health import assess_crop_health
from growth_guidance import RICE_GROWTH_STAGES, MAIZE_GROWTH_STAGES, estimate_current_stage
from yield_estimation import estimate_crop_yield

# -----------------------------------------------------------------------------
# Page Configuration & Clean Agricultural Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RiceMaize AI | Smart Agricultural Decision Support",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #2d6a4f;
        font-weight: 400;
        margin-bottom: 1.2rem;
        line-height: 1.5;
    }
    .metric-card {
        background: #f8faf8;
        border: 1px solid #d8f3dc;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.03);
    }
    .badge-healthy {
        background-color: #d8f3dc;
        color: #1b4332;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: #ffe6a7;
        color: #7f4f24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-danger {
        background-color: #ffd6d6;
        color: #9b2226;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Verified ICAR / TNAU / IRRI / CIMMYT Agricultural Knowledge Base
# -----------------------------------------------------------------------------
ADVISORY_DATABASE = {
    "Maize/Disease/01_maydis_leaf_blight": {
        "crop": "Maize (Corn)",
        "name": "Maydis Leaf Blight",
        "category": "Fungal Disease (Bipolaris maydis)",
        "severity": "High",
        "description": "Produces small, elongated, rectangular tan lesions bounded between veins. Rapidly blights foliage during warm, moist periods.",
        "organic": [
            "Spray <b>Trichoderma viride</b> (10 g/L) or <b>Pseudomonas fluorescens</b> (5 g/L) upon early symptom appearance.",
            "Apply <b>5% Neem Seed Kernel Extract (NSKE)</b> or neem oil (3 mL/L) with liquid soap emulsifier.",
            "Remove and compost infected lower leaves to restrict fungal spore splash.",
        ],
        "chemical": [
            "Foliar spray of <b>Mancozeb 75% WP</b> @ 2.5 g/L of water at early disease onset.",
            "In severe outbreaks, spray <b>Azoxystrobin 18.2% + Difenoconazole 11.4% SC</b> @ 1 mL/L of water.",
            "Repeat after 10–14 days if warm, humid weather persists.",
        ],
        "prevention": [
            "Avoid excessive single urea top-dressings which promote succulent, disease-prone leaves.",
            "Apply balanced Potash (MOP) to strengthen stalk and leaf cuticle cells.",
            "Plow under or compost crop residues immediately after harvest to destroy overwintering mycelium.",
        ],
    },
    "Maize/Disease/02_turcicum_leaf_blight": {
        "crop": "Maize (Corn)",
        "name": "Turcicum Leaf Blight",
        "category": "Fungal Disease (Exserohilum turcicum)",
        "severity": "High",
        "description": "Large, distinctive elliptical or cigar-shaped grayish-tan lesions on lower leaves that spread upward into the canopy.",
        "organic": [
            "Spray bio-fungicide <b>Trichoderma harzianum</b> (5 g/L) in compost tea during early morning.",
            "Apply fermented cow urine solution (diluted 1:10) as a natural immune stimulant.",
        ],
        "chemical": [
            "Spray <b>Carbendazim 12% + Mancozeb 63% WP</b> @ 2 g/L of water at first sign of cigar lesions.",
            "Alternatively, apply <b>Pyraclostrobin 20% WG</b> @ 1 g/L of water.",
        ],
        "prevention": [
            "Maintain proper spacing (60 cm × 20 cm) for air circulation and sunlight penetration.",
            "Avoid late evening overhead sprinkler irrigation that keeps leaves wet through the night.",
            "Deep summer plowing to bury fungal debris.",
        ],
    },
    "Maize/Disease/03_curvularia_leaf_spot": {
        "crop": "Maize (Corn)",
        "name": "Curvularia Leaf Spot",
        "category": "Fungal Disease (Curvularia lunata)",
        "severity": "Medium",
        "description": "Small, circular-to-oval chlorotic spots with brown centers and yellow halos across leaf blades.",
        "organic": [
            "Apply <b>Bacillus subtilis</b> bio-formulation (5 g/L) to colonize leaf surfaces.",
            "Foliar spray with neem oil (3 mL/L) with soap emulsifier.",
        ],
        "chemical": [
            "Spray <b>Chlorothalonil 75% WP</b> @ 2 g/L or <b>Propiconazole 25% EC</b> @ 1 mL/L.",
        ],
        "prevention": [
            "Incorporate well-rotted Farmyard Manure (FYM) to improve soil microbiology.",
            "Apply Zinc Sulphate (25 kg/ha) if deficiency is identified.",
            "Maintain unobstructed field drainage channels.",
        ],
    },
    "Maize/Disease/04_sorghum_downy_mildew": {
        "crop": "Maize (Corn)",
        "name": "Sorghum Downy Mildew",
        "category": "Oomycete / Downy Mildew (Peronosclerospora sorghi)",
        "severity": "Critical",
        "description": "Vivid yellow chlorotic striping along leaf veins, downy white fungal growth on lower leaf surfaces, and stunted 'crazy top' growth.",
        "organic": [
            "Immediately rogue out (uproot) and destroy systemically chlorotic plants to stop spore dispersal.",
            "Seed treatment with <b>Pseudomonas fluorescens</b> (10 g/kg seed) before sowing.",
        ],
        "chemical": [
            "Seed treatment with <b>Metalaxyl 35% WS</b> @ 4 g/kg seed before planting.",
            "Foliar spray with <b>Metalaxyl-M 4% + Mancozeb 64% WP</b> @ 2.5 g/L at 20 and 30 DAS.",
        ],
        "prevention": [
            "Rotate fields with pulses or oilseeds; avoid planting maize consecutively after sorghum.",
            "Avoid sowing in low-lying, poorly drained waterlogged plots.",
        ],
    },
    "Maize/Healthy": {
        "crop": "Maize (Corn)",
        "name": "Healthy Maize Crop",
        "category": "Optimal Health",
        "severity": "Healthy",
        "description": "Vigorous, dark green foliage with clean whorls and no evidence of fungal lesions, chlorosis, or insect chewing.",
        "organic": [
            "Apply vermicompost (2 tons/acre) and biofertilizers (Azospirillum & Phosphobacteria).",
            "Maintain pulse intercropping for natural soil nitrogen fixation.",
        ],
        "chemical": [
            "No chemical pesticides needed. Continue periodic scouting.",
        ],
        "prevention": [
            "Apply nitrogen in 3 splits (basal, knee-high at 30 DAS, and tasseling).",
            "Ensure steady soil moisture during critical tasseling and silking periods.",
            "Scout whorls weekly for early fall armyworm egg clusters.",
        ],
    },
    "Maize/Insect-pests/01_aphid": {
        "crop": "Maize (Corn)",
        "name": "Corn Leaf Aphid",
        "category": "Insect Pest (Rhopalosiphum maidis)",
        "severity": "Medium",
        "description": "Dense colonies of small bluish-green soft-bodied insects inside leaf whorls and tassels, sucking sap and secreting sticky honeydew.",
        "organic": [
            "Conserve natural predators: ladybird beetles (Coccinella) and green lacewings.",
            "Spray <b>5% Neem Seed Kernel Extract (NSKE)</b> or agricultural soap solution (5 mL/L).",
        ],
        "chemical": [
            "If >20% plants show heavy infestation, spray <b>Thiamethoxam 25% WG</b> @ 0.3 g/L or <b>Dimethoate 30% EC</b> @ 1.7 mL/L.",
        ],
        "prevention": [
            "Remove wild grasses and alternate weed hosts from field borders.",
            "Avoid excessive urea which produces tender tissues preferred by aphids.",
        ],
    },
    "Maize/Insect-pests/02_fall_armyworm": {
        "crop": "Maize (Corn)",
        "name": "Fall Armyworm (Larva)",
        "category": "Insect Pest (Spodoptera frugiperda)",
        "severity": "Critical",
        "description": "Voracious caterpillar with an inverted 'Y' on its head capsule and 4 square dots on segment 8. Chews leaves and bores into whorls and cobs.",
        "organic": [
            "Install pheromone traps @ 5/acre for adult moth monitoring.",
            "Release egg parasitoid <b>Trichogramma pretiosum</b> @ 50,000/ha at early emergence.",
            "Whorl application of <b>Metarhizium rileyi</b> or <b>Beauveria bassiana</b> (5 g/L).",
        ],
        "chemical": [
            "Direct nozzle spray into central whorls: <b>Chlorantraniliprole 18.5% SC</b> @ 0.4 mL/L.",
            "Or spray <b>Emamectin Benzoate 5% SG</b> @ 0.4 g/L or <b>Spinetoram 11.7% SC</b> @ 0.5 mL/L.",
        ],
        "prevention": [
            "Drop dry sand mixed with neem cake into central whorls to create an abrasive barrier for young larvae.",
            "Handpick egg masses during early seedling emergence (10–30 DAS).",
            "Synchronize sowing dates across adjacent plots to prevent continuous pest breeding.",
        ],
    },
    "Maize/Insect-pests/03_FAW_symptoms": {
        "crop": "Maize (Corn)",
        "name": "Fall Armyworm Damage Symptoms",
        "category": "Insect Pest Damage",
        "severity": "High",
        "description": "Characteristic window-paning (scraped papery leaf patches), ragged holes, and moist yellowish-brown sawdust-like frass inside the whorl.",
        "organic": [
            "Apply fine wood ash or sand mixed with lime (9:1) directly into affected leaf whorls.",
            "Spray <b>Bacillus thuringiensis (Bt kurstaki)</b> @ 2 g/L during late evening hours.",
        ],
        "chemical": [
            "Apply <b>Emamectin Benzoate 5% SG</b> @ 4 g/10L water directly into affected whorls.",
            "Alternate with <b>Chlorantraniliprole 18.5% SC</b> @ 4 mL/10L water to avoid resistance.",
        ],
        "prevention": [
            "Daily morning whorl inspection during the first 45 days after sowing.",
            "Install solar light traps @ 1 trap/acre to capture nocturnal moths.",
        ],
    },
    "Rice/Disease/01_Bacterial_leaf_blight": {
        "crop": "Rice (Paddy)",
        "name": "Bacterial Leaf Blight (BLB)",
        "category": "Bacterial Disease (Xanthomonas oryzae)",
        "severity": "High",
        "description": "Water-soaked stripes at leaf tips and wavy margins turning straw-yellow. Milky bacterial ooze droplets appear on lesions on humid mornings.",
        "organic": [
            "Spray <b>supernatant of fresh cow dung extract (20%)</b> or <b>Pseudomonas fluorescens</b> (5 g/L).",
            "Temporarily drain standing water for 3–4 days to arrest bacterial spread.",
        ],
        "chemical": [
            "Spray <b>Copper Hydroxide 77% WP</b> @ 2 g/L of water.",
            "Or combine <b>Streptocycline</b> (0.5 g/10L) with <b>Copper Oxychloride 50% WP</b> @ 25 g/10L of water.",
        ],
        "prevention": [
            "Strictly avoid urea top-dressing during cloudy or stormy weather.",
            "Apply potash in two splits (50% basal + 50% at panicle initiation) to strengthen leaf tissue.",
            "Maintain shallow 2–3 cm standing water.",
        ],
    },
    "Rice/Disease/02_Brown_spot": {
        "crop": "Rice (Paddy)",
        "name": "Brown Spot Disease",
        "category": "Fungal Disease (Helminthosporium oryzae)",
        "severity": "Medium",
        "description": "Brown, circular-to-oval spots with gray centers and yellow halos resembling sesame seeds across leaf blades.",
        "organic": [
            "Seed biopriming with <b>Trichoderma asperellum</b> (10 g/kg seed) before nursery sowing.",
            "Foliar spray with <b>5% Neem Seed Kernel Extract (NSKE)</b>.",
        ],
        "chemical": [
            "Spray <b>Tricyclazole 75% WP</b> @ 0.6 g/L or <b>Propiconazole 25% EC</b> @ 1 mL/L.",
            "Or apply <b>Mancozeb 75% WP</b> @ 2 g/L at tillering and boot leaf stage.",
        ],
        "prevention": [
            "Correct soil nutrient deficiencies: ensure balanced N-P-K (120:40:40 kg/ha).",
            "Apply <b>Zinc Sulphate @ 25 kg/ha</b> at basal stage.",
            "Incorporate green manure (Dhaincha/Sunn hemp) before puddling.",
        ],
    },
    "Rice/Disease/03_False_smut": {
        "crop": "Rice (Paddy)",
        "name": "False Smut Disease",
        "category": "Fungal Disease (Ustilaginoidea virens)",
        "severity": "High",
        "description": "Transforms individual rice grains into large velvety green-to-orange powdery spore balls, turning dark greenish-black at maturity.",
        "organic": [
            "Foliar spray with fermented cow urine + ginger-garlic extract (5%) prior to panicle emergence.",
            "Carefully handpick and bag smut balls from affected panicles to prevent field contamination.",
        ],
        "chemical": [
            "Preventive spray at boot stage: <b>Copper Oxychloride 50% WP</b> @ 2.5 g/L.",
            "Or spray <b>Tebuconazole 25.9% EC</b> @ 1.5 mL/L or <b>Azoxystrobin + Difenoconazole</b> @ 1 mL/L.",
        ],
        "prevention": [
            "Avoid excessive or late nitrogen application after panicle initiation.",
            "Moderate water drainage during grain filling phase.",
            "Burn or deeply bury infected panicles after harvest.",
        ],
    },
    "Rice/Disease/04_leaf_sheath_blight": {
        "crop": "Rice (Paddy)",
        "name": "Sheath Blight",
        "category": "Fungal Disease (Rhizoctonia solani)",
        "severity": "High",
        "description": "Oval, grayish-green water-soaked lesions on lower leaf sheaths near the water level. Lesions enlarge with dark brown borders and cause lodging.",
        "organic": [
            "Apply <b>Trichoderma viride-enriched FYM</b> (2.5 kg in 50 kg manure per acre) at transplanting.",
            "Spray <b>Pseudomonas fluorescens</b> (10 g/L) targeting lower stem canopy.",
        ],
        "chemical": [
            "Spray <b>Hexaconazole 5% SC</b> @ 2 mL/L or <b>Validamycin 3% L</b> @ 2 mL/L of water.",
            "Direct chemical spray toward the lower leaf sheaths near the water level.",
        ],
        "prevention": [
            "Avoid over-dense transplanting; keep spacing at least 20 cm × 15 cm.",
            "Keep field bunds and irrigation canals free from wild weed hosts.",
            "Drain standing water intermittently to lower relative humidity in the canopy.",
        ],
    },
    "Rice/Healthy": {
        "crop": "Rice (Paddy)",
        "name": "Healthy Rice Crop",
        "category": "Optimal Health",
        "severity": "Healthy",
        "description": "Clean, vibrant green upright tillers with no boreholes, folded leaves, or fungal lesions.",
        "organic": [
            "Incorporate Azolla biofertilizer into paddy water (fixes nitrogen and suppresses weeds).",
            "Use Pseudomonas fluorescens root dip at seedling transplanting.",
        ],
        "chemical": [
            "No chemical pesticides needed. Continue regular field monitoring.",
        ],
        "prevention": [
            "Adopt Alternate Wetting and Drying (AWD) irrigation: saves water and strengthens root anchors.",
            "Apply recommended potash split to enhance stem strength against lodging.",
            "Scout for stem borer egg masses on leaf tips every 5–7 days.",
        ],
    },
    "Rice/Insect-pests/05_Leaf_folder": {
        "crop": "Rice (Paddy)",
        "name": "Rice Leaf Folder",
        "category": "Insect Pest (Cnaphalocrocis medinalis)",
        "severity": "High",
        "description": "Larva folds leaf edges longitudinally with silk threads, scraping green chlorophyll from inside to leave papery white streaks.",
        "organic": [
            "Release egg parasitoid <b>Trichogramma chilonis</b> @ 100,000/ha at 7-day intervals.",
            "Pull a long coir rope across the crop canopy back and forth to dislodge larvae into flood water.",
            "Spray <b>5% Neem Seed Kernel Extract (NSKE)</b>.",
        ],
        "chemical": [
            "If >10% leaves are folded at vegetative stage, spray <b>Chlorantraniliprole 18.5% SC</b> @ 0.3 mL/L.",
            "Or apply <b>Cartap Hydrochloride 50% SP</b> @ 2 g/L or <b>Flubendiamide 39.35% SC</b> @ 0.2 mL/L.",
        ],
        "prevention": [
            "Avoid over-application of urea which creates tender leaves favored by leaf folder moths.",
            "Clear shaded field edges and weed hosts where moths shelter during daytime.",
        ],
    },
    "Rice/Insect-pests/06_Rice_skipper": {
        "crop": "Rice (Paddy)",
        "name": "Rice Skipper",
        "category": "Insect Pest (Pelopidas mathias)",
        "severity": "Medium",
        "description": "Green caterpillars that roll single leaf blades into tubular shelters, emerging to chew leaf margins and defoliate young tillers.",
        "organic": [
            "Manually collect and destroy rolled leaf tips containing caterpillars or pupae.",
            "Install bird perches (T-shaped bamboo sticks, 10/acre) to encourage insectivorous birds.",
            "Spray <b>Beauveria bassiana</b> @ 5 g/L during cool evening hours.",
        ],
        "chemical": [
            "If defoliation exceeds threshold (>1 larva/hill), spray <b>Quinalphos 25% EC</b> @ 2 mL/L.",
        ],
        "prevention": [
            "Trim grassy weeds on bunds (Echinochloa species) that serve as alternate host plants.",
            "Maintain proper water levels so dropped larvae drown.",
        ],
    },
    "Rice/Insect-pests/07_White_stem_borer": {
        "crop": "Rice (Paddy)",
        "name": "White Stem Borer",
        "category": "Insect Pest (Scirpophaga innotata)",
        "severity": "Critical",
        "description": "Larva bores into tiller bases, causing 'Dead Hearts' (wilting central shoot) during tillering and 'White Earhead' (bleached empty panicle) at heading.",
        "organic": [
            "Install sex pheromone traps @ 5 traps/acre for adult moth monitoring and mass trapping.",
            "Clip the top 2 cm of seedling leaf tips before transplanting to remove deposited egg masses.",
            "Conserve natural larval parasitoids (Tetrastichus and Bracon species).",
        ],
        "chemical": [
            "Broadcast <b>Chlorantraniliprole 0.4% GR</b> @ 4 kg/acre or <b>Cartap Hydrochloride 4% GR</b> @ 7.5 kg/acre into standing water.",
            "Or spray <b>Fipronil 5% SC</b> @ 2 mL/L of water at threshold (1 egg mass/sq. meter).",
        ],
        "prevention": [
            "Synchronize transplanting with neighboring fields to break the pest generation cycle.",
            "Harvest rice close to ground level and plow stubbles immediately to destroy diapausing larvae.",
        ],
    },
    "Rice/Insect-pests/08_Yellow_stem_borer": {
        "crop": "Rice (Paddy)",
        "name": "Yellow Stem Borer",
        "category": "Insect Pest (Scirpophaga incertulas)",
        "severity": "Critical",
        "description": "Destructive pest causing 'Dead Hearts' in vegetative stage and 'Whiteheads' with empty panicles at heading.",
        "organic": [
            "Install pheromone traps with <b>Scirpolure septa</b> @ 5 traps/acre.",
            "Install light traps (1/acre) to attract and destroy night-flying female moths.",
            "Release egg parasitoid <b>Trichogramma japonicum</b> @ 100,000/ha at weekly intervals.",
        ],
        "chemical": [
            "Apply <b>Cartap Hydrochloride 50% SP</b> @ 2 g/L or <b>Chlorantraniliprole 18.5% SC</b> @ 0.3 mL/L.",
            "Or broadcast <b>Fipronil 0.3% GR</b> @ 10 kg/acre into standing water.",
        ],
        "prevention": [
            "Clip seedling tips before transplanting to discard hidden egg clusters.",
            "Submerge or burn stubbles after harvest to eliminate overwintering caterpillars.",
        ],
    },
}

# -----------------------------------------------------------------------------
# Cached Model Loader (Uses Top Performing Trained ViT)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_model():
    class_names = load_class_names()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("vit", num_classes=len(class_names), pretrained=False)
    checkpoint_path = ROOT_DIR / "checkpoints" / "vit.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Trained weights not found in {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, class_names, device


# -----------------------------------------------------------------------------
# Header — 2-Line Overview
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🌾 RiceMaize AI — Smart Crop Health & Decision Support</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Empowering farmers with AI-powered crop diagnostics, visual explanations, and verified agronomic support.<br>'
    'Upload a crop photo to receive instant diagnosis, reliability analysis, treatment solutions, and growth guidance.'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_diag, tab_health, tab_growth, tab_yield, tab_guide = st.tabs(
    [
        "📸 Instant Diagnosis & Explainable AI",
        "🩺 Crop Health Assessment",
        "🌱 Growth Stage Agronomy",
        "📊 Yield & Production Estimator",
        "📖 Field Guide & Library",
    ]
)

# =============================================================================
# TAB 1: INSTANT CROP DIAGNOSIS & EXPLAINABLE AI
# =============================================================================
with tab_diag:
    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("### 📤 Upload Crop Photo")
        st.write("Upload a clear, well-lit photo of the affected plant leaf, stem, or pest.")

        uploaded_file = st.file_uploader(
            "Choose a leaf or pest photo",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supported formats: JPG, PNG, WEBP",
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("#### 🧪 Or Test With a Sample Photo")

        sample_options = {
            "Select a sample image...": None,
            "🌾 Rice - Healthy Leaf": ROOT_DIR / "Dataset/Rice/Healthy/0c60af21-42a0-4e69-b957-a101c6a7d5c0.Jpg",
            "🌾 Rice - Brown Spot Disease": ROOT_DIR / "Dataset/Rice/Disease/02_Brown_spot/02_Brown_spot (1).jpg",
            "🌾 Rice - Bacterial Leaf Blight": ROOT_DIR / "Dataset/Rice/Disease/01_Bacterial_leaf_blight/01_Bacterial_leaf_blight (1).jpg",
            "🌽 Maize - Healthy Leaf": ROOT_DIR / "Dataset/Maize/Healthy/0a1a6b09-b68e-4a62-b90d-27b0eb8cb111.Jpg",
            "🌽 Maize - Fall Armyworm Symptoms": ROOT_DIR / "Dataset/Maize/Insect-pests/03_FAW_symptoms/0a2ff911-3957-41a4-927a-85d7f3001899.Jpg",
            "🌽 Maize - Maydis Leaf Blight": ROOT_DIR / "Dataset/Maize/Disease/01_maydis_leaf_blight/01b33a7e-ada2-4963-a011-b6fbb6486912.Jpg",
        }

        selected_sample_label = st.selectbox(
            "Choose sample:",
            list(sample_options.keys()),
            label_visibility="collapsed",
        )
        selected_sample_path = sample_options[selected_sample_label]

        active_image = None

        if uploaded_file is not None:
            active_image = Image.open(uploaded_file).convert("RGB")
        elif selected_sample_path and Path(selected_sample_path).exists():
            active_image = Image.open(selected_sample_path).convert("RGB")

        if active_image is not None:
            # Feature 3: Image Quality Validation
            quality_rep = evaluate_image_quality(active_image)

            if quality_rep.status == "Good Quality":
                st.success(f"✅ **Image Quality**: {quality_rep.status} (Sharpness: {quality_rep.metrics['sharpness_laplacian']:.0f})")
            elif quality_rep.status == "Borderline / Quality Warning":
                st.warning(f"⚠️ **Image Quality Notice**: {quality_rep.status}")
                with st.expander("View Quality Suggestions"):
                    for iss in quality_rep.issues:
                        st.write(f"• {iss}")
                    for tip in quality_rep.recommendations:
                        st.info(f"💡 {tip}")
            else:
                st.error(f"❌ **Image Quality Unsuitable**: {quality_rep.status}")
                for iss in quality_rep.issues:
                    st.write(f"• {iss}")
                for tip in quality_rep.recommendations:
                    st.warning(f"💡 {tip}")

    with col_result:
        st.markdown("### 📋 Diagnostic Result & Reliability")

        if active_image is None:
            st.info("👈 **Upload an image or pick a sample photo** on the left to begin diagnosis.")
            st.markdown(
                """
                <div style="background-color: #f8faf8; border-radius: 10px; padding: 20px; border: 1px dashed #b7e4c7; color: #52796f;">
                    <b>💡 Tips for Best Diagnostic Accuracy:</b>
                    <ul>
                        <li>Capture the diseased leaf blade or pest clearly in diffuse natural daylight.</li>
                        <li>Avoid blurry photos; hold camera steady 15–20 cm away.</li>
                        <li>Keep both healthy and diseased margins visible on the leaf.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analyzing image features and evaluating reliability..."):
                model, class_names, device = get_model()
                tensor = preprocess_pil(active_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)

                # Feature 2: Reliability & Validation Temperature Calibration
                rel_rep = assess_prediction_reliability(
                    logits=logits,
                    class_names=class_names,
                    temperature=VALIDATION_TEMPERATURE,
                    top_k=3,
                )

                raw_class = rel_rep.predicted_class
                advisory = ADVISORY_DATABASE.get(
                    raw_class,
                    {
                        "crop": raw_class.split("/")[0],
                        "name": raw_class.split("/")[-1].replace("_", " ").title(),
                        "category": "Agricultural Condition",
                        "severity": "Medium",
                        "description": "Identified by agricultural visual recognition.",
                        "organic": ["Apply neem oil spray and maintain soil aeration."],
                        "chemical": ["Consult local agricultural officer for specific advice."],
                        "prevention": ["Maintain balanced NPK fertilization and clean drainage."],
                    },
                )

                # Feature 1: Explainable AI Attribution
                explained_img, xai_caption = explain_prediction(
                    model=model,
                    model_type="vit",
                    tensor=tensor,
                    original_image=active_image,
                    target_class=class_names.index(raw_class),
                )

            # Badge styling
            if advisory["severity"] == "Healthy":
                badge_html = '<span class="badge-healthy">🟢 HEALTHY CROP — VIGOROUS</span>'
            elif advisory["severity"] == "Critical":
                badge_html = '<span class="badge-danger">🔴 CRITICAL ACTION REQUIRED</span>'
            else:
                badge_html = '<span class="badge-warning">🟠 ATTENTION REQUIRED — ACTIVE CONDITION</span>'

            # Result card
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 1.15rem; font-weight: 600; color: #1b4332;">{advisory['crop']}</span>
                        {badge_html}
                    </div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #081c15; margin-bottom: 4px;">
                        {advisory['name']}
                    </div>
                    <div style="font-size: 0.95rem; color: #40916c; font-weight: 500; margin-bottom: 8px;">
                        Nature: {advisory['category']}
                    </div>
                    <div style="font-size: 0.9rem; color: #2d6a4f; margin-bottom: 4px;">
                        <b>Calibrated Confidence:</b> {rel_rep.calibrated_confidence:.1f}% &nbsp;•&nbsp; 
                        <b>Reliability Status:</b> <i>{rel_rep.reliability_level}</i>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(rel_rep.calibrated_confidence / 100.0, 1.0))

            # Feature 2: Cautionary Notice / Uncertainty Warning
            if rel_rep.warning_notice:
                st.warning(rel_rep.warning_notice)

            # Feature 2: Top-3 candidate predictions
            with st.expander("📊 View Top-3 Model Candidates (Calibrated Probabilities)"):
                for rank, (cand_name, cand_prob) in enumerate(rel_rep.top_predictions, start=1):
                    cand_display = cand_name.split("/")[-1].replace("_", " ")
                    st.write(f"**{rank}. {cand_name.split('/')[0]} — {cand_display}**: `{cand_prob:.1f}%`")

            # Feature 1: Side-by-side Visual Explanation
            st.markdown("#### 🔍 Explainable AI (Visual Focus Heatmap)")
            col_orig, col_xai = st.columns(2)
            with col_orig:
                st.image(active_image, caption="Original Uploaded Photo", use_container_width=True)
            with col_xai:
                if explained_img is not None:
                    st.image(explained_img, caption="AI Attribution Heatmap (Warm = Strongest Focus)", use_container_width=True)
                else:
                    st.caption("Visual explanation could not be generated.")

            st.caption(
                "ℹ️ **Interpretability Notice**: Warm red/yellow highlighted zones indicate leaf areas that most influenced "
                "the model's prediction. This visualization aids human verification and does not constitute microbiological proof."
            )

            # Feature 4: Verified Treatment & Advisory Tabs
            treat_tab1, treat_tab2, treat_tab3 = st.tabs(
                ["🌿 Organic & Biological Care", "🧪 Chemical Control", "🚜 Prevention & Field Recovery"]
            )

            with treat_tab1:
                st.markdown("##### 🌿 Organic, Cultural & Biological Remedies")
                st.write("Safe for soil ecology, pollinators, and beneficial insects:")
                for item in advisory["organic"]:
                    st.markdown(f"• {item}", unsafe_allow_html=True)

            with treat_tab2:
                st.markdown("##### 🧪 Recommended Chemical Treatments")
                st.write("Targeted options for severe infections or threshold-exceeding outbreaks:")
                for item in advisory["chemical"]:
                    st.markdown(f"• {item}", unsafe_allow_html=True)
                st.warning("⚠️ **Safety Notice**: Adhere to manufacturer label dosages, pre-harvest intervals (PHI), and wear protective gloves/masks.")

            with treat_tab3:
                st.markdown("##### 🚜 Agronomic Recovery & Prevention")
                st.write("Prevent pathogen recurrence and enhance plant resilience:")
                for item in advisory["prevention"]:
                    st.markdown(f"• {item}", unsafe_allow_html=True)


# =============================================================================
# TAB 2: CROP HEALTH ASSESSMENT (FEATURE 5)
# =============================================================================
with tab_health:
    st.markdown("### 🩺 Multi-Factor Crop Health Assessment")
    st.write(
        "Synthesize visual diagnostic findings with your real-time field observations "
        "(crop stage, field spread percentage, and microclimate conditions) to assess overall crop health risk."
    )

    col_h_in1, col_h_in2 = st.columns(2)

    with col_h_in1:
        h_crop = st.selectbox("Select Crop Type:", ["Rice (Paddy)", "Maize (Corn)"], key="h_crop")
        h_condition = st.selectbox(
            "Current Observed / Diagnosed Condition:",
            list(ADVISORY_DATABASE.keys()),
            format_func=lambda x: f"{x.split('/')[0]} — {ADVISORY_DATABASE[x]['name']}",
            key="h_cond",
        )
        h_confidence = st.slider("Diagnostic Confidence (%):", min_value=10, max_value=100, value=85, step=5)

    with col_h_in2:
        h_stage = st.selectbox(
            "Current Crop Growth Stage:",
            [
                "Seedling / Nursery",
                "Active Vegetative / Tillering",
                "Panicle Initiation / Knee-High",
                "Flowering / Heading / Tasseling",
                "Grain / Cob Filling",
                "Maturity",
            ],
            key="h_stage",
        )
        h_spread = st.selectbox(
            "Estimated Field Area Affected:",
            [
                "Isolated / Few Plants (<5%)",
                "Patchy / Localized Spots (5% – 20%)",
                "Widespread / Severe (>20%)",
            ],
            key="h_spread",
        )
        h_weather = st.selectbox(
            "Recent Weather Over Past 7 Days:",
            [
                "Moderate / Normal Sunlight",
                "Heavy Rain / High Humidity / Cloud Cover",
                "Hot & Dry / Water Stress",
            ],
            key="h_weather",
        )

    if st.button("Generate Crop Health Assessment Report", type="primary"):
        health_rep = assess_crop_health(
            predicted_class=h_condition,
            confidence=float(h_confidence),
            crop_stage=h_stage,
            spread_level=h_spread,
            weather_condition=h_weather,
        )

        st.markdown("---")
        st.markdown("#### 📑 Field Health Evaluation Summary")

        risk_color = "#1b4332" if health_rep.risk_level == "Low" else ("#e63946" if health_rep.risk_level in ("High", "Critical") else "#f77f00")

        st.markdown(
            f"""
            <div style="background-color: #f8faf8; border-left: 6px solid {risk_color}; padding: 18px 24px; border-radius: 8px; margin-bottom: 16px;">
                <h4 style="margin: 0 0 6px 0; color: {risk_color};">Status: {health_rep.health_status} (Risk Level: {health_rep.risk_level})</h4>
                <p style="margin: 0; color: #2d6a4f; font-size: 1.05rem;">{health_rep.summary}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown("##### 🔎 Contributing Field Observations")
            for f in health_rep.contributing_factors:
                st.write(f"• {f}")
        with col_f2:
            st.markdown("##### 📋 Recommended Next Steps")
            for a in health_rep.recommended_actions:
                st.write(f"• {a}")

        st.caption(f"ℹ️ {health_rep.limitations_disclaimer}")


# =============================================================================
# TAB 3: GROWTH STAGE AGRONOMY (FEATURE 6)
# =============================================================================
with tab_growth:
    st.markdown("### 🌱 Crop-Stage Agronomy & Growth Guidance")
    st.write("Verified stage-by-stage agronomic advisories based on ICAR, TNAU, and IRRI best practices.")

    g_col1, g_col2 = st.columns([1, 1.5], gap="large")

    with g_col1:
        st.markdown("#### 📅 Stage Calculator")
        g_crop = st.radio("Choose Crop:", ["Rice (Paddy)", "Maize (Corn)"], horizontal=True)
        sow_date = st.date_input("Sowing / Transplanting Date:", value=date.today())

        current_stage = estimate_current_stage(g_crop, sow_date)
        if current_stage is not None:
            st.success(f"🌾 **Estimated Stage**: {current_stage.stage_name}")
            st.caption(f"Estimated timeframe: {current_stage.duration_days}")

    with g_col2:
        st.markdown(f"#### 📖 Detailed Agronomy Guide ({g_crop})")
        stages_list = RICE_GROWTH_STAGES if "Rice" in g_crop else MAIZE_GROWTH_STAGES

        for stage in stages_list:
            is_active = (current_stage is not None and stage.stage_name == current_stage.stage_name)
            expander_title = f"{'⭐ [CURRENT] ' if is_active else ''}{stage.stage_name} ({stage.duration_days})"

            with st.expander(expander_title, expanded=is_active):
                st.markdown(f"**💧 Irrigation & Water Management:**<br>{stage.irrigation_advice}", unsafe_allow_html=True)
                st.markdown(f"**🌱 Nutrient & Fertilizer Scheduling:**<br>{stage.nutrient_management}", unsafe_allow_html=True)
                st.markdown(f"**🌿 Weed Management:**<br>{stage.weed_control}", unsafe_allow_html=True)
                st.markdown(f"**🐛 Pest & Disease Vigilance:**<br>{stage.pest_disease_watch}", unsafe_allow_html=True)
                st.markdown("**⭐ Good Agricultural Practices:**")
                for bp in stage.best_practices:
                    st.write(f"• {bp}")


# =============================================================================
# TAB 4: CROP YIELD & PRODUCTION ESTIMATOR (FEATURE 7)
# =============================================================================
with tab_yield:
    st.markdown("### 📊 Crop Yield & Production Estimator (ML Regression)")
    st.write(
        "A separate machine learning regression module trained on historical district-level agricultural data "
        "to estimate expected yield (Tonnes/Ha) and total production (Tonnes) based on agro-climatic inputs."
    )

    y_col1, y_col2 = st.columns(2)

    with y_col1:
        y_state = st.selectbox(
            "State / Agro-Climatic Region:",
            ["Punjab", "Tamil Nadu", "Andhra Pradesh", "Telangana", "Karnataka", "West Bengal", "Bihar", "Uttar Pradesh", "Madhya Pradesh", "Maharashtra"],
        )
        y_crop = st.selectbox("Crop:", ["Rice", "Maize"])
        y_season = st.selectbox("Season:", ["Kharif", "Rabi", "Summer"])
        y_area = st.number_input("Cultivated Area (Hectares):", min_value=0.1, max_value=50000.0, value=5.0, step=0.5)

    with y_col2:
        y_rainfall = st.slider("Seasonal Rainfall (mm):", min_value=150.0, max_value=2500.0, value=850.0, step=25.0)
        y_fertilizer = st.slider("Fertilizer Application (Total NPK kg/ha):", min_value=40.0, max_value=300.0, value=140.0, step=10.0)
        y_irrig = st.slider("Area Under Assured Irrigation (%):", min_value=10.0, max_value=100.0, value=85.0, step=5.0)

    if st.button("Calculate Yield & Production Estimate", type="primary"):
        with st.spinner("Running regression model..."):
            y_res = estimate_crop_yield(
                state=y_state,
                crop=y_crop,
                season=y_season,
                area_hectares=float(y_area),
                annual_rainfall_mm=float(y_rainfall),
                fertilizer_npk_kg_ha=float(y_fertilizer),
                irrigation_pct=float(y_irrig),
            )

        st.markdown("---")
        st.markdown("#### 📈 Model Estimation Output")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                label="Estimated Yield per Hectare",
                value=f"{y_res.predicted_yield_per_ha:.2f} T/ha",
                help=f"Expected 90% range: {y_res.yield_lower_bound:.2f} – {y_res.yield_upper_bound:.2f} Tonnes/Hectare",
            )
            st.caption(f"Confidence Range: `{y_res.yield_lower_bound:.2f} – {y_res.yield_upper_bound:.2f} T/ha`")

        with m2:
            st.metric(
                label=f"Total Production ({y_res.area_hectares} ha)",
                value=f"{y_res.predicted_total_production:,.1f} Tonnes",
                help=f"Expected 90% range: {y_res.production_lower_bound:,.1f} – {y_res.production_upper_bound:,.1f} Tonnes",
            )
            st.caption(f"Production Range: `{y_res.production_lower_bound:,.1f} – {y_res.production_upper_bound:,.1f} Tonnes`")

        with m3:
            st.metric(
                label="Regression Model R²",
                value=f"{y_res.model_r2_score:.3f}",
                help=f"Test Set MAE: {y_res.model_mae:.3f} Tonnes/Hectare",
            )
            st.caption(f"Test MAE: `{y_res.model_mae:.3f} T/ha`")

        st.info(f"ℹ️ {y_res.limitations_note}")


# =============================================================================
# TAB 5: FIELD GUIDE & DISEASE LIBRARY
# =============================================================================
with tab_guide:
    st.markdown("### 📖 Farmer's Crop Condition Field Guide")
    st.write("Browse reference descriptions, symptoms, and verified solutions for all 17 Rice and Maize conditions.")

    filter_c1, filter_c2 = st.columns(2)
    with filter_c1:
        crop_sel = st.selectbox("Filter by Crop:", ["All Crops", "Rice (Paddy)", "Maize (Corn)"], key="fg_crop")
    with filter_c2:
        cat_sel = st.selectbox("Filter by Category:", ["All Categories", "Fungal Disease", "Bacterial Disease", "Insect Pest", "Optimal Health"], key="fg_cat")

    st.markdown("---")

    filtered_guide = []
    for k, v in ADVISORY_DATABASE.items():
        if crop_sel != "All Crops":
            pfx = "Rice" if "Rice" in crop_sel else "Maize"
            if not v["crop"].startswith(pfx):
                continue
        if cat_sel != "All Categories":
            if cat_sel not in v["category"]:
                continue
        filtered_guide.append((k, v))

    for k, v in filtered_guide:
        icon = "🟢" if v["severity"] == "Healthy" else ("🔴" if v["severity"] == "Critical" else "🟠")
        with st.expander(f"{icon} {v['crop']} — {v['name']} ({v['category']})"):
            st.markdown(f"**Symptoms & Nature:** {v['description']}")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("##### 🌿 Organic / Biological Care")
                for o in v["organic"]:
                    st.markdown(f"- {o}", unsafe_allow_html=True)
            with c2:
                st.markdown("##### 🧪 Chemical Control")
                for c in v["chemical"]:
                    st.markdown(f"- {c}", unsafe_allow_html=True)
            st.markdown("##### 🚜 Prevention & Health Recovery")
            for p in v["prevention"]:
                st.markdown(f"- {p}", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        """
        <div style="background-color: #edf6f9; border-radius: 10px; padding: 18px 24px; border-left: 5px solid #0077b6;">
            <h4 style="color: #03045e; margin-top: 0;">📞 Emergency Farmer Advisory & Assistance</h4>
            <p style="margin-bottom: 6px; color: #023e8a;">
                If disease outbreaks or pest infestations exceed economic threshold in your area, contact your local Krishi Vigyan Kendra (KVK) or the National Kisan Call Center:
            </p>
            <b style="font-size: 1.1rem; color: #03045e;">Toll-Free Kisan Call Center: 1800-180-1551</b> (Operational 6:00 AM to 10:00 PM daily)
        </div>
        """,
        unsafe_allow_html=True,
    )
