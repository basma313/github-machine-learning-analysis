
import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import plotly.express as px

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="wide"
)

CLASSES = [
    "Actinic Keratoses",
    "Basal Cell Carcinoma",
    "Benign Keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions"
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "dermascan_final.keras"
    )

st.markdown("""
<style>
.stApp {
    background-color: #F8F7FC;
}
h1, h2, h3 {
    color: #392D59;
}
.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #49366F;
}
.subtitle {
    color: #77718B;
    font-size: 18px;
}
div.stButton > button {
    background-color: #7355A5;
    color: white;
    border-radius: 12px;
    border: none;
    padding: 12px 28px;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #594080;
    color: white;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">DermaScan AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered educational skin image analysis</div>',
    unsafe_allow_html=True
)

st.divider()

st.header("Skin Image Analysis")

st.write(
    "Upload a skin image to explore the model's predictions."
)

uploaded_file = st.file_uploader(
    "Upload your image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=350
    )

    if st.button("Analyze Image"):
        try:
            with st.spinner("Analyzing image..."):
                model = load_model()

                resized = image.resize((224, 224))
                image_array = np.array(resized)
                image_array = np.expand_dims(
                    image_array, axis=0
                )

                predictions = model.predict(
                    image_array,
                    verbose=0
                )[0]

                if len(predictions) != len(CLASSES):
                    st.error("Unexpected model output.")
                    st.stop()

                results = pd.DataFrame({
                    "Category": CLASSES,
                    "Probability": predictions * 100
                })

                results = results.sort_values(
                    "Probability",
                    ascending=False
                )

                st.success("Analysis completed!")

                st.subheader("Prediction Results")

                st.metric(
                    "Top predicted category",
                    results.iloc[0]["Category"]
                )

                st.caption(
                    "This is the model's highest-scoring category, "
                    "not a confirmed diagnosis."
                )

                st.subheader("Category Probabilities")

                st.dataframe(
                    results.style.format({
                        "Probability": "{:.2f}%"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                fig = px.bar(
                    results,
                    x="Probability",
                    y="Category",
                    orientation="h",
                    title="Model Prediction Scores",
                    color="Probability",
                    color_continuous_scale="Purples"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        except Exception as error:
            st.error(
                "Analysis could not be completed. "
                "Please try another image."
            )
            st.exception(error)

st.divider()

st.subheader("Important Medical Notice")

st.warning(
    "DermaScan AI is an educational research project. "
    "It is not a medical diagnostic tool and cannot confirm "
    "or rule out any skin condition. Model predictions may "
    "be incorrect. Consult a qualified healthcare professional "
    "for medical concerns."
)

st.caption("DermaScan AI | Educational AI Project")
