
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
* {
    box-sizing: border-box;
}

.stApp {
    background: #F8FAFC;
    color: #19496F;
}

header[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    max-width: 1400px;
    padding: 2rem 4% 2rem 4%;
}

h1, h2, h3 {
    color: #19496F !important;
}

p {
    color: #45596B;
}

.main-header {
    background: white;
    padding: 20px 28px;
    border-radius: 15px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 35px;
    box-shadow: 0 3px 15px rgba(0,0,0,0.03);
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
}

.brand-icon {
    font-size: 35px;
    color: #19496F;
}

.brand-name {
    font-size: 25px;
    font-weight: 800;
    color: #19496F;
}

.brand-subtitle {
    font-size: 13px;
    color: #718096;
}

.header-link {
    background: #19496F;
    color: white !important;
    padding: 13px 22px;
    border-radius: 8px;
    text-decoration: none !important;
    font-weight: bold;
}

.hero-tag {
    display: inline-block;
    background: #E4F0F4;
    color: #19496F;
    padding: 11px 15px;
    border-radius: 7px;
    font-size: 12px;
    font-weight: bold;
    margin-bottom: 15px;
}

.hero-title {
    font-size: clamp(38px, 5vw, 55px);
    font-weight: 800;
    line-height: 1.15;
    color: #19496F;
    margin: 15px 0 20px 0;
}

.hero-description {
    font-size: 19px;
    line-height: 1.6;
    color: #45596B;
    margin-bottom: 25px;
}

.visual-box {
    background: #E8B7A7;
    border-radius: 25px;
    min-height: 310px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #19496F;
    font-size: 20px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
}

.visual-caption {
    color: #718096;
    font-size: 13px;
    text-align: center;
    margin-top: 10px;
}

.section-title {
    text-align: center;
    font-size: 30px;
    font-weight: 800;
    margin-top: 35px;
}

.section-subtitle {
    text-align: center;
    color: #718096;
    margin-bottom: 25px;
}

.step-card {
    background: #EAF2F6;
    border-radius: 15px;
    padding: 25px;
    min-height: 190px;
}

.step-number {
    background: #438F91;
    color: white;
    width: 43px;
    height: 43px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    font-weight: bold;
}

.step-title {
    font-size: 19px;
    font-weight: bold;
    color: #19496F;
    margin-top: 15px;
    margin-bottom: 8px;
}

.step-description {
    font-size: 14px;
    line-height: 1.5;
    color: #718096;
}

div.stButton > button {
    background: #19496F;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: bold;
    min-height: 48px;
}

div.stButton > button:hover {
    background: #123752;
    color: white;
}

div[data-testid="stFileUploader"] {
    background: white;
    border: 1px dashed #438F91;
    border-radius: 12px;
    padding: 12px;
}

div[data-testid="stFileUploader"] button {
    background: #19496F;
    color: white;
}

.result-card {
    background: white;
    padding: 22px;
    border-radius: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    margin: 15px 0;
}

.notice {
    background: #EAF2F6;
    border-left: 5px solid #438F91;
    border-radius: 8px;
    padding: 18px;
    color: #45596B;
    font-size: 13px;
    line-height: 1.6;
    margin-top: 25px;
}

.footer {
    text-align: center;
    color: #718096;
    font-size: 12px;
    margin-top: 35px;
    padding: 15px;
}

[data-testid="stMetricValue"] {
    color: #19496F;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 5%;
    }

    .hero-title {
        font-size: 38px;
    }
}
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown("""
<div class="main-header">
    <div class="brand">
        <div class="brand-icon">✚</div>
        <div>
            <div class="brand-name">DermaScan AI</div>
            <div class="brand-subtitle">
                Know Your Skin. Care Better.
            </div>
        </div>
    </div>
    <a href="#upload-section" class="header-link">
        Get Started →
    </a>
</div>
""", unsafe_allow_html=True)


# HERO SECTION

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("""
    <div class="hero-tag">
        ✦ AI-POWERED SKIN SCREENING
    </div>

    <div class="hero-title">
        Upload.<br>
        Detect.<br>
        Care.
    </div>

    <div class="hero-description">
        Explore possible skin conditions with AI-assisted
        image analysis and clear, helpful information.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div id="upload-section"></div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a Skin Image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        st.success("Image uploaded successfully!")

        analyze = st.button(
            "✦ Analyze Image",
            use_container_width=True
        )
    else:
        analyze = False
        st.info("Upload an image to start the analysis.")


with right:
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Your Uploaded Skin Image",
            use_container_width=True
        )
    else:
        st.markdown("""
        <div class="visual-box">
            <div>
                ✚<br><br>
                Your Skin Image<br>
                Will Appear Here
            </div>
        </div>
        <div class="visual-caption">
            Upload an image to preview it here
        </div>
        """, unsafe_allow_html=True)


# MODEL ANALYSIS

if uploaded_file and analyze:

    try:
        with st.spinner("Analyzing your image..."):

            model = load_model()

            image = Image.open(uploaded_file).convert("RGB")
            resized = image.resize((224, 224))

            image_array = np.array(resized)
            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            predictions = model.predict(
                image_array,
                verbose=0
            )[0]

            if len(predictions) != len(CLASSES):
                raise ValueError("Unexpected model output.")

            results = pd.DataFrame({
                "Category": CLASSES,
                "Probability": predictions * 100
            })

            results = results.sort_values(
                "Probability",
                ascending=False
            ).reset_index(drop=True)

            st.session_state["dermascan_results"] = results

    except Exception:
        st.error(
            "Analysis could not be completed. "
            "Please try another image."
        )
        st.session_state.pop("dermascan_results", None)


# RESULTS

if "dermascan_results" in st.session_state:

    results = st.session_state["dermascan_results"]

    st.divider()

    st.markdown(
        '<div class="section-title">Analysis Results</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-subtitle">'
        'AI-assisted image analysis results'
        '</div>',
        unsafe_allow_html=True
    )

    top_category = results.iloc[0]["Category"]
    top_score = results.iloc[0]["Probability"]

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Top Predicted Category",
            top_category
        )

    with col2:
        st.metric(
            "Model Score",
            f"{top_score:.2f}%"
        )

    st.caption(
        "The highest-scoring category is a model prediction, "
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
        results.sort_values(
            "Probability",
            ascending=True
        ),
        x="Probability",
        y="Category",
        orientation="h",
        title="Model Prediction Scores",
        color_discrete_sequence=["#19496F"]
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#19496F",
        xaxis_title="Model Score (%)",
        yaxis_title="",
        showlegend=False
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# HOW IT WORKS

st.divider()

st.markdown(
    '<div class="section-title">How It Works</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">'
    'Three simple steps'
    '</div>',
    unsafe_allow_html=True
)

step1, step2, step3 = st.columns(3, gap="medium")

with step1:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">01</div>
        <div class="step-title">Upload an Image</div>
        <div class="step-description">
            Choose a clear photo of the skin area.
        </div>
    </div>
    """, unsafe_allow_html=True)

with step2:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">02</div>
        <div class="step-title">AI-Assisted Analysis</div>
        <div class="step-description">
            The AI model analyzes the uploaded image
            and generates prediction scores.
        </div>
    </div>
    """, unsafe_allow_html=True)

with step3:
    st.markdown("""
    <div class="step-card">
        <div class="step-number">03</div>
        <div class="step-title">Explore Information</div>
        <div class="step-description">
            Explore the model's possible categories
            and their prediction scores.
        </div>
    </div>
    """, unsafe_allow_html=True)


# MEDICAL DISCLAIMER

st.markdown("""
<div class="notice">
    <strong>Important Medical Notice</strong><br><br>
    DermaScan AI is an educational research project.
    It is not a medical diagnostic tool and cannot confirm
    or rule out any skin condition. Model predictions may
    be incorrect. Consult a qualified healthcare professional
    for medical concerns.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    DermaScan AI | Know Your Skin. Care Better.
</div>
""", unsafe_allow_html=True)
