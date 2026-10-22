import sys
from pathlib import Path

# Ensure project root is in sys.path even when running from frontend/ directory
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

# -----------------------------------------------------------------------------
# Page Configuration & Professional Agricultural Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="RiceMaize AI | Smart Crop Health & Diagnostic System",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Modern, clean typography & agricultural green palette */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1b4332;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
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
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: #ffe6a7;
        color: #7f4f24;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-danger {
        background-color: #ffd6d6;
        color: #9b2226;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .treatment-box {
        background-color: #ffffff;
        border-left: 4px solid #2d6a4f;
        padding: 14px 18px;
        border-radius: 8px;
        margin-top: 10px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Advisory & Treatment Database for All 17 Rice & Maize Conditions
# -----------------------------------------------------------------------------
ADVISORY_DATABASE = {
    "Maize/Disease/01_maydis_leaf_blight": {
        "crop": "Maize (Corn)",
        "name": "Maydis Leaf Blight",
        "category": "Fungal Disease",
        "severity": "High",
        "description": "Fungal infection caused by Bipolaris maydis. Produces small, elongated, rectangular tan lesions between leaf veins. In humid conditions, lesions expand rapidly, scorching foliage and limiting grain filling.",
        "organic": [
            "Foliar spray of <b>Trichoderma viride</b> or <b>Pseudomonas fluorescens</b> (10 g/L of water) at the first sight of lesions.",
            "Apply <b>5% Neem Seed Kernel Extract (NSKE)</b> or neem oil (3 mL/L) with soap emulsifier every 10 days.",
            "Remove and compost infected lower foliage to reduce local fungal spore load.",
        ],
        "chemical": [
            "Spray <b>Mancozeb 75% WP</b> @ 2.5 g/L of water at early disease onset.",
            "In severe cases, spray <b>Azoxystrobin 18.2% + Difenoconazole 11.4% SC</b> @ 1 mL/L of water.",
            "Repeat application after 10–14 days if wet, warm weather continues.",
        ],
        "prevention": [
            "Avoid excessive urea/nitrogen applications which create lush, vulnerable tissues.",
            "Apply adequate potash (K) to reinforce stalk and leaf epidermal cell walls.",
            "Practice crop rotation with non-cereal crops (soybeans, pulses) for at least 1–2 seasons.",
            "Plow down or destroy infected maize stubbles after harvest to destroy overwintering fungal mycelium.",
        ],
    },
    "Maize/Disease/02_turcicum_leaf_blight": {
        "crop": "Maize (Corn)",
        "name": "Turcicum Leaf Blight",
        "category": "Fungal Disease",
        "severity": "High",
        "description": "Also known as Northern Corn Leaf Blight, caused by Exserohilum turcicum. Recognized by large, distinct cigar-shaped grayish-tan lesions that appear first on lower leaves before advancing upward.",
        "organic": [
            "Spray bio-fungicide <b>Trichoderma harzianum</b> (5 g/L) mixed with compost tea in the early morning.",
            "Apply fermented cow urine solution (diluted 1:10) as a natural immune booster and fungal retardant.",
        ],
        "chemical": [
            "Spray <b>Carbendazim 12% + Mancozeb 63% WP</b> @ 2 g/L of water.",
            "Alternatively, apply <b>Pyraclostrobin 20% WG</b> @ 1 g/L of water upon observing the first cigar-shaped lesion.",
        ],
        "prevention": [
            "Maintain recommended plant spacing (60 cm × 20 cm) to allow sunlight penetration and airflow.",
            "Avoid overhead sprinkler irrigation late in the evening to prevent prolonged leaf wetness.",
            "Deep summer plowing to bury infested plant debris.",
        ],
    },
    "Maize/Disease/03_curvularia_leaf_spot": {
        "crop": "Maize (Corn)",
        "name": "Curvularia Leaf Spot",
        "category": "Fungal Disease",
        "severity": "Medium",
        "description": "Fungal leaf spotting caused by Curvularia lunata. Produces small, circular-to-oval spots with yellow halos and dark brown centers, primarily on intermediate and upper leaves.",
        "organic": [
            "Spray <b>Bacillus subtilis</b> bio-formulation (5 g/L) to colonize leaf surfaces.",
            "Apply <b>neem oil (3 mL/L)</b> mixed with liquid soap as an organic preventive film.",
        ],
        "chemical": [
            "Spray <b>Chlorothalonil 75% WP</b> @ 2 g/L or <b>Propiconazole 25% EC</b> @ 1 mL/L.",
        ],
        "prevention": [
            "Enrich soil with well-rotted Farmyard Manure (FYM) to improve root health.",
            "Apply Zinc Sulphate (25 kg/ha) if zinc deficiency is observed in the field.",
            "Ensure field channels prevent stagnant water pools.",
        ],
    },
    "Maize/Disease/04_sorghum_downy_mildew": {
        "crop": "Maize (Corn)",
        "name": "Sorghum Downy Mildew",
        "category": "Fungal Disease",
        "severity": "Critical",
        "description": "Caused by Peronosclerospora sorghi. Causes vivid chlorotic (yellowish) striping on leaves, white downy fungal growth on leaf undersides during cool humid mornings, and stunted plant growth ('crazy top').",
        "organic": [
            "Immediately rogue out (uproot) and burn systemically infected chlorotic plants to protect the rest of the field.",
            "Apply <b>Pseudomonas fluorescens</b> seed treatment before planting.",
        ],
        "chemical": [
            "Seed treatment with <b>Metalaxyl 35% WS</b> @ 4 g/kg seed before sowing.",
            "Foliar spray with <b>Metalaxyl-M 4% + Mancozeb 64% WP</b> @ 2.5 g/L at 20 and 30 days after sowing.",
        ],
        "prevention": [
            "Strictly rotate fields with pulses or vegetables; avoid planting maize consecutively after sorghum.",
            "Ensure deep field drainage; avoid low-lying waterlogged patches.",
        ],
    },
    "Maize/Healthy": {
        "crop": "Maize (Corn)",
        "name": "Healthy Maize Crop",
        "category": "Optimal Health",
        "severity": "Healthy",
        "description": "Vibrant, deep-green foliage with no signs of fungal lesions, leaf chlorosis, chewing damage, or stalk decay. The plant exhibits robust photosynthetic vigor.",
        "organic": [
            "Continue applying vermicompost (2 tons/acre) and biofertilizers (Azospirillum & Phosphobacteria).",
            "Maintain intercropping with leguminous crops (cowpea or soybean) for natural nitrogen enrichment.",
        ],
        "chemical": [
            "No chemical pesticides needed. Maintain regular scouting.",
        ],
        "prevention": [
            "Split nitrogen application: 1/3 at sowing, 1/3 at knee-high stage (30 DAS), and 1/3 at tasseling.",
            "Maintain consistent soil moisture during critical tasseling and cob filling stages.",
            "Scout leaf whorls weekly to catch any early signs of fall armyworm egg masses.",
        ],
    },
    "Maize/Insect-pests/01_aphid": {
        "crop": "Maize (Corn)",
        "name": "Corn Leaf Aphid",
        "category": "Insect Pest",
        "severity": "Medium",
        "description": "Tiny bluish-green soft-bodied insects clustering inside the leaf whorl, upper leaves, and tassels. They suck sap, weaken developing tassels, and secrete sticky honeydew that fosters black sooty mold.",
        "organic": [
            "Conserve or release predatory ladybird beetles (Coccinella septempunctata) and green lacewings.",
            "Foliar spray with <b>5% Neem Seed Kernel Extract (NSKE)</b> or agricultural soap solution (5 mL/L).",
        ],
        "chemical": [
            "If severe (>20% plants infested), spray <b>Thiamethoxam 25% WG</b> @ 0.3 g/L or <b>Dimethoate 30% EC</b> @ 1.7 mL/L.",
        ],
        "prevention": [
            "Clear wild grasses and weed hosts from field boundaries.",
            "Avoid excessive synthetic nitrogen which causes succulent growth preferred by aphid colonies.",
        ],
    },
    "Maize/Insect-pests/02_fall_armyworm": {
        "crop": "Maize (Corn)",
        "name": "Fall Armyworm (Larva)",
        "category": "Insect Pest",
        "severity": "Critical",
        "description": "Highly destructive caterpillar with an inverted 'Y' on its head capsule and 4 square dots on the 8th segment. Devours whorl leaves, cuts tassels, and bores directly into growing cobs.",
        "organic": [
            "Install pheromone traps @ 5 traps/acre for early male moth detection.",
            "Release egg parasitoid <b>Trichogramma pretiosum</b> @ 50,000/ha at early crop stages.",
            "Whorl application of <b>Metarhizium rileyi</b> or <b>Beauveria bassiana</b> (5 g/L).",
            "Push-pull intercropping with Desmodium repels moths, while Napier grass borders trap them.",
        ],
        "chemical": [
            "Direct nozzle spray into the central whorl with <b>Chlorantraniliprole 18.5% SC</b> @ 0.4 mL/L.",
            "Alternatively, spray <b>Emamectin Benzoate 5% SG</b> @ 0.4 g/L or <b>Spinetoram 11.7% SC</b> @ 0.5 mL/L.",
        ],
        "prevention": [
            "Handpick and crush egg masses and young larvae during early crop emergence (10–30 DAS).",
            "Drop dry sand mixed with neem cake into central whorls to create an abrasive barrier for larvae.",
            "Synchronize planting dates with neighboring fields to prevent staggered pest breeding.",
        ],
    },
    "Maize/Insect-pests/03_FAW_symptoms": {
        "crop": "Maize (Corn)",
        "name": "Fall Armyworm Damage Symptoms",
        "category": "Insect Pest Damage",
        "severity": "High",
        "description": "Characteristic window-paning (scraped papery leaf patches), ragged chewed leaf holes, and dense yellowish-brown sawdust-like frass inside the central leaf whorl.",
        "organic": [
            "Apply fine wood ash or sand mixed with lime (9:1) directly into affected leaf whorls.",
            "Spray <b>Bacillus thuringiensis (Bt kurstaki)</b> @ 2 g/L during late evening hours.",
        ],
        "chemical": [
            "Apply <b>Emamectin Benzoate 5% SG</b> @ 4 g / 10L water directed straight into the whorls.",
            "Alternate with <b>Chlorantraniliprole 18.5% SC</b> @ 4 mL / 10L water to prevent chemical resistance.",
        ],
        "prevention": [
            "Perform daily morning whorl scouting during the first 45 days after germination.",
            "Install solar light traps @ 1 trap/acre to capture adult night-flying moths.",
        ],
    },
    "Rice/Disease/01_Bacterial_leaf_blight": {
        "crop": "Rice (Paddy)",
        "name": "Bacterial Leaf Blight (BLB)",
        "category": "Bacterial Disease",
        "severity": "High",
        "description": "Caused by Xanthomonas oryzae. Lesions begin as water-soaked stripes at leaf tips and margins, turning yellow to straw-colored with wavy edges. In humid mornings, milky bacterial ooze beads on lesion surfaces.",
        "organic": [
            "Spray <b>supernatant of fresh cow dung extract (20%)</b> or <b>Pseudomonas fluorescens</b> (5 g/L).",
            "Temporarily drain standing water from the field for 3–4 days to stop bacterial transmission.",
        ],
        "chemical": [
            "Spray <b>Copper Hydroxide 77% WP</b> @ 2 g/L of water.",
            "Or combine <b>Streptocycline</b> (0.5 g/10L) with <b>Copper Oxychloride 50% WP</b> @ 25 g/10L of water.",
        ],
        "prevention": [
            "Strictly avoid excessive urea top-dressing during cloudy, humid, or rainy spells.",
            "Apply potash in two splits (50% basal + 50% at panicle initiation) to strengthen leaf tissue.",
            "Avoid deep standing water; maintain shallow 2–3 cm depth.",
        ],
    },
    "Rice/Disease/02_Brown_spot": {
        "crop": "Rice (Paddy)",
        "name": "Brown Spot Disease",
        "category": "Fungal Disease",
        "severity": "Medium",
        "description": "Caused by Helminthosporium oryzae. Produces brown, circular-to-oval spots with yellow haloes resembling sesame seeds across leaves and glumes. Strongly associated with nutrient-deficient or poor soils.",
        "organic": [
            "Seed biopriming with <b>Trichoderma asperellum</b> (10 g/kg seed) before nursery sowing.",
            "Spray <b>5% Neem Seed Kernel Extract (NSKE)</b> or sour fermented buttermilk (50 mL/L).",
        ],
        "chemical": [
            "Spray <b>Tricyclazole 75% WP</b> @ 0.6 g/L or <b>Propiconazole 25% EC</b> @ 1 mL/L.",
            "Or apply <b>Mancozeb 75% WP</b> @ 2 g/L at tillering and boot leaf stages.",
        ],
        "prevention": [
            "Correct soil nutrient deficiencies: ensure balanced N-P-K (120:40:40 kg/ha).",
            "Apply <b>Zinc Sulphate @ 25 kg/ha</b> at basal stage.",
            "Incorporate green manure (Dhaincha/Sunn hemp) or farmyard manure before puddling.",
        ],
    },
    "Rice/Disease/03_False_smut": {
        "crop": "Rice (Paddy)",
        "name": "False Smut Disease",
        "category": "Fungal Disease",
        "severity": "High",
        "description": "Caused by Ustilaginoidea virens. Transforms developing grains into velvety orange-yellow or dark greenish-black powdery smut balls, reducing grain weight and milling quality.",
        "organic": [
            "Foliar spray with fermented cow urine + ginger-garlic extract (5%) prior to panicle emergence.",
            "Carefully handpick and bag smut balls from affected panicles to prevent field contamination.",
        ],
        "chemical": [
            "Preventive spray at boot stage (prior to 50% flowering): <b>Copper Oxychloride 50% WP</b> @ 2.5 g/L.",
            "Or spray <b>Tebuconazole 25.9% EC</b> @ 1.5 mL/L or <b>Azoxystrobin + Difenoconazole</b> @ 1 mL/L.",
        ],
        "prevention": [
            "Avoid excessive or late nitrogen top-dressing after panicle initiation.",
            "Moderate water drainage during grain filling phase.",
            "Burn or deeply bury infected panicles after harvest.",
        ],
    },
    "Rice/Disease/04_leaf_sheath_blight": {
        "crop": "Rice (Paddy)",
        "name": "Sheath Blight",
        "category": "Fungal Disease",
        "severity": "High",
        "description": "Caused by Rhizoctonia solani. Oval, grayish-green water-soaked lesions appear on lower leaf sheaths near the water surface. Lesions enlarge and coalesce, spreading up into leaves and causing lodging.",
        "organic": [
            "Apply <b>Trichoderma viride-enriched FYM</b> (2.5 kg in 50 kg manure per acre) at transplanting.",
            "Spray <b>Pseudomonas fluorescens</b> (10 g/L) directly targeting the lower stem canopy.",
        ],
        "chemical": [
            "Spray <b>Hexaconazole 5% SC</b> @ 2 mL/L or <b>Validamycin 3% L</b> @ 2 mL/L of water.",
            "Direct the chemical spray toward the lower leaf sheaths near the water level.",
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
        "description": "Vigorous tillers with clean leaf sheaths, rich green upright leaves, and completely free of stem borer boreholes, folded leaves, or discoloration spots.",
        "organic": [
            "Incorporate Azolla biofertilizer into paddy water (fixes nitrogen and suppresses weeds).",
            "Use Pseudomonas fluorescens root dip at the seedling transplanting stage.",
        ],
        "chemical": [
            "No chemical pesticides needed. Continue periodic field monitoring.",
        ],
        "prevention": [
            "Adopt Alternate Wetting and Drying (AWD) irrigation: save up to 30% water and promote deep roots.",
            "Apply recommended potash split to enhance stem strength against lodging.",
            "Scout for stem borer egg masses on leaf tips every 5–7 days.",
        ],
    },
    "Rice/Insect-pests/05_Leaf_folder": {
        "crop": "Rice (Paddy)",
        "name": "Rice Leaf Folder",
        "category": "Insect Pest",
        "severity": "High",
        "description": "Larva folds leaf edges together with fine silk threads, staying protected inside and scraping green chlorophyll, leaving behind transparent whitish papery streaks.",
        "organic": [
            "Release egg parasitoid <b>Trichogramma chilonis</b> @ 100,000/ha at 7-day intervals.",
            "Pull a long coir rope across the crop canopy back and forth to dislodge larvae into flood water.",
            "Spray <b>5% Neem Seed Kernel Extract (NSKE)</b>.",
        ],
        "chemical": [
            "If >10% leaves are folded during vegetative stage, spray <b>Chlorantraniliprole 18.5% SC</b> @ 0.3 mL/L.",
            "Or apply <b>Cartap Hydrochloride 50% SP</b> @ 2 g/L or <b>Flubendiamide 39.35% SC</b> @ 0.2 mL/L.",
        ],
        "prevention": [
            "Avoid over-application of urea; excess nitrogen causes tender leaves favored by leaf folder moths.",
            "Clear shaded field edges and weed hosts where moths take shelter during daytime.",
        ],
    },
    "Rice/Insect-pests/06_Rice_skipper": {
        "crop": "Rice (Paddy)",
        "name": "Rice Skipper",
        "category": "Insect Pest",
        "severity": "Medium",
        "description": "Green caterpillars that roll single leaf blades or bend leaf tips into tubular chambers, emerging to chew leaf margins and defoliate young tillers.",
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
        "category": "Insect Pest",
        "severity": "Critical",
        "description": "Larva bores into the base of the tiller and feeds internally, causing 'Dead Heart' (wilting central shoot) during vegetative stage and 'White Earhead' (empty bleached panicle) at heading stage.",
        "organic": [
            "Install sex pheromone traps @ 5 traps/acre for adult moth monitoring and mass trapping.",
            "Clip the top 2 inches of seedling leaf tips before transplanting to remove deposited egg masses.",
            "Conserve natural larval parasitoids (Tetrastichus and Bracon species).",
        ],
        "chemical": [
            "Broadcast <b>Chlorantraniliprole 0.4% GR</b> @ 4 kg/acre or <b>Cartap Hydrochloride 4% GR</b> @ 7.5 kg/acre into 2–3 cm standing water.",
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
        "category": "Insect Pest",
        "severity": "Critical",
        "description": "Destructive monophagous pest. Larvae bore into stems resulting in 'Dead Hearts' during vegetative tillering and 'Whiteheads' with chaffy, grainless panicles during heading.",
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
# Cached Model Loader & Predictor (Uses Top Performing Trained ViT)
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


def diagnose_image(image: Image.Image):
    model, class_names, device = get_model()
    tensor = preprocess_pil(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    confidence, pred_idx = probs.max(dim=0)
    raw_class = class_names[int(pred_idx)]
    conf_score = float(confidence) * 100.0

    advisory = ADVISORY_DATABASE.get(
        raw_class,
        {
            "crop": raw_class.split("/")[0],
            "name": raw_class.split("/")[-1].replace("_", " ").title(),
            "category": "Unknown",
            "severity": "Medium",
            "description": "Condition detected by visual analysis.",
            "organic": ["Apply neem oil spray and maintain soil health."],
            "chemical": ["Consult local agricultural officer for specific chemical advice."],
            "prevention": ["Maintain balanced NPK fertilization and clean drainage."],
        },
    )

    return advisory, conf_score, raw_class


# -----------------------------------------------------------------------------
# Header — 2-Line Farmer-Centric Overview
# -----------------------------------------------------------------------------
st.markdown('<div class="main-title">🌾 RiceMaize AI — Smart Crop Health & Diagnostic System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Empowering farmers with AI-powered crop health diagnostics for Rice and Maize.<br>'
    'Upload a field photo of your crop leaf or pest to receive instant diagnosis, treatment solutions, and recovery recommendations.'
    '</div>',
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Main Navigation Tabs
# -----------------------------------------------------------------------------
tab_diag, tab_guide, tab_tips = st.tabs(
    ["📸 Instant Crop Diagnosis", "📖 Field Guide & Disease Library", "🌱 Agronomy & Best Practices"]
)

# =============================================================================
# TAB 1: INSTANT CROP DIAGNOSIS
# =============================================================================
with tab_diag:
    col_input, col_result = st.columns([1, 1.2], gap="large")

    with col_input:
        st.markdown("### 📤 Upload Crop Photo")
        st.write("Take a clear, well-lit photo of the affected leaf, stem, or pest.")

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
            st.image(active_image, caption="Uploaded Field Photo", use_container_width=True)
        elif selected_sample_path and Path(selected_sample_path).exists():
            active_image = Image.open(selected_sample_path).convert("RGB")
            st.image(active_image, caption=f"Sample: {selected_sample_label}", use_container_width=True)

    with col_result:
        st.markdown("### 📋 Diagnostic Result & Action Plan")

        if active_image is None:
            st.info(
                "👈 **Please upload a crop leaf image** or select one of the sample photos on the left to view the diagnosis and treatment advice."
            )
            st.markdown(
                """
                <div style="background-color: #f8faf8; border-radius: 10px; padding: 20px; border: 1px dashed #b7e4c7; color: #52796f;">
                    <b>💡 Tips for Best Diagnostic Accuracy:</b>
                    <ul>
                        <li>Take close-up photos focusing directly on the leaf spots, lesions, or insects.</li>
                        <li>Ensure natural daylight and avoid intense direct glare or heavy shadows.</li>
                        <li>Avoid blurry photos; keep the phone camera steady before clicking.</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            with st.spinner("Analyzing leaf symptoms and matching agricultural database..."):
                advisory, conf, raw_class = diagnose_image(active_image)

            # Determine badge styling based on severity
            if advisory["severity"] == "Healthy":
                badge_html = '<span class="badge-healthy">🟢 HEALTHY CROP — VIGOROUS</span>'
            elif advisory["severity"] == "Critical":
                badge_html = '<span class="badge-danger">🔴 CRITICAL ACTION REQUIRED</span>'
            else:
                badge_html = '<span class="badge-warning">🟠 ATTENTION REQUIRED — ACTIVE CONDITION</span>'

            # Diagnostic summary card (NO raw models, NO raw debug data)
            st.markdown(
                f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 1.1rem; font-weight: 600; color: #1b4332;">{advisory['crop']}</span>
                        {badge_html}
                    </div>
                    <div style="font-size: 1.6rem; font-weight: 700; color: #081c15; margin-bottom: 4px;">
                        {advisory['name']}
                    </div>
                    <div style="font-size: 0.95rem; color: #40916c; font-weight: 500; margin-bottom: 12px;">
                        Category: {advisory['category']}
                    </div>
                    <div style="margin-bottom: 6px; font-size: 0.9rem; color: #2d6a4f;">
                        <b>Diagnostic Confidence:</b> {conf:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(min(conf / 100.0, 1.0))

            st.markdown(f"**Condition Overview:** {advisory['description']}")

            # Treatment and Improvement Tabs
            treat_tab1, treat_tab2, treat_tab3 = st.tabs(
                ["🌿 Organic & Biological Remedies", "🧪 Chemical Treatment", "🚜 Health Improvement & Prevention"]
            )

            with treat_tab1:
                st.markdown("#### 🌿 Eco-Friendly & Biological Solutions")
                st.write("Safe for soil, beneficial pollinators, and human health:")
                for item in advisory["organic"]:
                    st.markdown(f"• {item}", unsafe_allow_html=True)

            with treat_tab2:
                st.markdown("#### 🧪 Targeted Chemical Treatments")
                st.write("Recommended if condition is severe or spreading across field plots:")
                for item in advisory["chemical"]:
                    st.markdown(f"• {item}", unsafe_allow_html=True)
                st.caption("⚠️ Always adhere to safety labels, wear protective gear, and follow pre-harvest intervals.")

            with treat_tab3:
                st.markdown("#### 🚜 Plant Health & Soil Recovery Suggestions")
                st.write("Actions to strengthen crop immunity and prevent recurring outbreaks:")
                for item in advisory["prevention"]:
                    st.markdown(f"• {item}", unsafe_allow_html=True)


# =============================================================================
# TAB 2: FIELD GUIDE & DISEASE LIBRARY
# =============================================================================
with tab_guide:
    st.markdown("### 📖 Farmer's Crop Condition Field Guide")
    st.write("Browse reference descriptions, identification symptoms, and solutions for all 17 Rice and Maize conditions.")

    filter_col1, filter_col2 = st.columns([1, 1])
    with filter_col1:
        crop_filter = st.selectbox("Filter by Crop:", ["All Crops", "Rice (Paddy)", "Maize (Corn)"])
    with filter_col2:
        cat_filter = st.selectbox("Filter by Category:", ["All Categories", "Fungal Disease", "Bacterial Disease", "Insect Pest", "Optimal Health"])

    st.markdown("---")

    filtered_items = []
    for key, data in ADVISORY_DATABASE.items():
        if crop_filter != "All Crops":
            target_prefix = "Rice" if "Rice" in crop_filter else "Maize"
            if not data["crop"].startswith(target_prefix):
                continue
        if cat_filter != "All Categories":
            if cat_filter not in data["category"]:
                continue
        filtered_items.append((key, data))

    if not filtered_items:
        st.warning("No conditions match your selected filters.")
    else:
        for key, data in filtered_items:
            severity_icon = "🟢" if data["severity"] == "Healthy" else ("🔴" if data["severity"] == "Critical" else "🟠")
            with st.expander(f"{severity_icon} {data['crop']} — {data['name']} ({data['category']})"):
                st.markdown(f"**Symptoms & Nature:** {data['description']}")
                
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("##### 🌿 Organic / Biological Care")
                    for o in data["organic"]:
                        st.markdown(f"- {o}", unsafe_allow_html=True)
                with c2:
                    st.markdown("##### 🧪 Chemical Control")
                    for c in data["chemical"]:
                        st.markdown(f"- {c}", unsafe_allow_html=True)

                st.markdown("##### 🚜 Prevention & Health Improvement")
                for p in data["prevention"]:
                    st.markdown(f"- {p}", unsafe_allow_html=True)


# =============================================================================
# TAB 3: AGRONOMY & BEST PRACTICES
# =============================================================================
with tab_tips:
    st.markdown("### 🌱 Good Agricultural Practices (GAP)")
    st.write("General guidelines for maintaining robust crop immunity, reducing input costs, and preventing disease outbreaks.")

    col_ag1, col_ag2 = st.columns(2, gap="medium")

    with col_ag1:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color: #1b4332; margin-top: 0;">🌾 Rice Agronomy Essentials</h4>
                <ul>
                    <li><b>Alternate Wetting & Drying (AWD):</b> Rather than continuous deep flooding, allow water to naturally subside before re-irrigating. This saves 25–30% water and aerates root zones, reducing sheath blight and root rot.</li>
                    <li><b>Balanced Potash (K):</b> Potash strengthens the stem cuticle, making it harder for stem borers to penetrate and preventing crop lodging during maturity.</li>
                    <li><b>Prevent Nitrogen Spikes:</b> Heavy single doses of urea trigger sudden vegetative flushes with soft cell walls, heavily encouraging Leaf Blast and Bacterial Leaf Blight. Split nitrogen into 3–4 applications.</li>
                    <li><b>Seedling Tip Clipping:</b> Trimming the top 2 cm of rice seedlings prior to transplanting removes over 80% of yellow and white stem borer egg clusters deposited in nurseries.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_ag2:
        st.markdown(
            """
            <div class="metric-card">
                <h4 style="color: #1b4332; margin-top: 0;">🌽 Maize Agronomy Essentials</h4>
                <ul>
                    <li><b>Optimal Plant Spacing:</b> Maintain 60 cm row-to-row and 20 cm plant-to-plant distance. Overcrowded fields create a humid microclimate that triggers Maydis and Turcicum leaf blights.</li>
                    <li><b>Early Whorl Scouting for FAW:</b> Inspect 20 plants across 5 spots in the field weekly between 10 to 40 days after emergence. Look for window-paning on young leaves.</li>
                    <li><b>Intercropping Benefits:</b> Intercropping maize with pulses (cowpea, black gram) fixes atmospheric nitrogen, suppresses weeds, and breaks the life cycle of specialized maize pests.</li>
                    <li><b>Critical Moisture Stages:</b> Water stress during tasseling and silking directly hampers pollination and ear formation. Prioritize irrigation during these two weeks.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        """
        <div style="background-color: #edf6f9; border-radius: 10px; padding: 18px 24px; border-left: 5px solid #0077b6;">
            <h4 style="color: #03045e; margin-top: 0;">📞 Emergency Farmer Advisory & Assistance</h4>
            <p style="margin-bottom: 6px; color: #023e8a;">
                If disease outbreaks or pest infestations exceed 20% of your field area, contact your nearest Krishi Vigyan Kendra (KVK) or the National Kisan Call Center:
            </p>
            <b style="font-size: 1.1rem; color: #03045e;">Toll-Free Kisan Call Center: 1800-180-1551</b> (Operational 6:00 AM to 10:00 PM daily)
        </div>
        """,
        unsafe_allow_html=True,
    )
