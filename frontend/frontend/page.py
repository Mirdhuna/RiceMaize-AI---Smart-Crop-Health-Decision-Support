import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="RiceMaize AI",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #f4f8f1;
}
h1, h2, h3 {
    color: #205c35;
}
.stButton > button {
    background-color: #287a43;
    color: white;
    border-radius: 10px;
    height: 45px;
}
</style>
""", unsafe_allow_html=True)

st.title("🌱 RiceMaize AI")
st.subheader("AI-Powered Crop Disease Detection")

st.write(
    "Upload a rice or maize leaf image to begin crop health analysis."
)

crop = st.selectbox("Select your crop", ["Rice", "Maize"])

uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded leaf", width=400)

    if st.button("Check Disease", use_container_width=True):
        st.info(
            "Image uploaded successfully. "
            "The prediction model will be connected next."
        )

st.divider()

st.caption(
    "RiceMaize AI | Machine Learning Based Agricultural Support"
)
