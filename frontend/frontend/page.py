import streamlit as st
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
PREDICT_SCRIPT = ROOT / "predict.py"

st.set_page_config(
    page_title="RiceMaize AI",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f3f8f0;
}
h1, h2, h3 {
    color: #205c35;
}
.stButton > button {
    background-color: #287a43;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 RiceMaize AI")
st.subheader("AI-Powered Crop Disease Detection")

st.write(
    "Upload a rice or maize leaf image to identify its "
    "predicted health category using machine learning."
)

st.divider()

left, right = st.columns(2, gap="large")

with left:
    st.header("📸 Upload Leaf Image")

    crop = st.selectbox(
        "Select crop",
        ["Rice", "Maize"]
    )

    model_name = st.selectbox(
        "Select prediction model",
        ["hybrid", "cnn", "vit"],
        format_func=lambda x: {
            "hybrid": "Hybrid (CNN + ViT)",
            "cnn": "CNN",
            "vit": "Vision Transformer (ViT)"
        }[x]
    )

    uploaded_file = st.file_uploader(
        "Choose a leaf image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(
            image,
            caption=f"{crop} leaf image",
            use_container_width=True
        )

with right:
    st.header("🔍 Disease Detection")

    st.write(
        "Upload a clear leaf image and click the button "
        "to run your trained model."
    )

    if st.button("Check Crop Health", use_container_width=True):
        if uploaded_file is None:
            st.warning("Please upload a leaf image first.")

        elif not PREDICT_SCRIPT.exists():
            st.error("predict.py was not found in the project root.")

        else:
            temp_path = None

            try:
                suffix = Path(uploaded_file.name).suffix.lower()

                with tempfile.NamedTemporaryFile(
                    suffix=suffix,
                    delete=False
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = Path(temp_file.name)

                command = [
                    sys.executable,
                    str(PREDICT_SCRIPT),
                    str(temp_path),
                    "--model", model_name,
                    "--checkpoints", str(ROOT / "checkpoints"),
                    "--top-k", "3"
                ]

                with st.spinner("Analyzing leaf image..."):
                    result = subprocess.run(
                        command,
                        cwd=str(ROOT),
                        capture_output=True,
                        text=True,
                        timeout=180
                    )

                if result.returncode != 0:
                    st.error("Prediction could not be completed.")
                    st.code(result.stderr or result.stdout)

                else:
                    output = result.stdout.strip()
                    values = {}

                    for line in output.splitlines():
                        if ":" in line:
                            key, value = line.split(":", 1)
                            values[key.strip()] = value.strip()

                    st.success("Prediction completed!")

                    st.markdown("### 🌿 Prediction Result")

                    st.write(
                        f"**Selected crop:** {crop}"
                    )
                    st.write(
                        f"**Model used:** {values.get('Model', model_name)}"
                    )
                    st.write(
                        f"**Predicted category:** "
                        f"{values.get('Predicted class', 'Unavailable')}"
                    )

                    confidence = values.get("Confidence")

                    if confidence:
                        st.metric(
                            "Model confidence",
                            confidence
                        )

                    st.markdown("### 📊 Top Predictions")

                    if "Top predictions:" in output:
                        top_predictions = output.split(
                            "Top predictions:", 1
                        )[1].strip()

                        if top_predictions:
                            st.code(top_predictions)

                    st.info(
                        "This is a model prediction, not a guaranteed "
                        "diagnosis. Verify results before choosing "
                        "crop treatment."
                    )

            except subprocess.TimeoutExpired:
                st.error("Prediction timed out. Please try again.")

            except Exception as error:
                st.error(f"An error occurred: {error}")

            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

st.divider()

st.markdown("""
###About RiceMaize AI

RiceMaize AI combines machine learning-based crop image
classification with the goal of supporting better agricultural
decision-making.

**Note:** Model results depend on the trained model and its
training data. Always confirm important crop-management
decisions with qualified agricultural experts.
""")
