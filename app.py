
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="centered"
)

st.title("DermaScan AI")
st.write("AI-based Skin Image Classification")

st.markdown("---")

st.header("Upload a Skin Image")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze Image"):
        st.info("Model connection is coming next!")

st.markdown("---")
st.caption("Educational project only — not a medical diagnosis.")
