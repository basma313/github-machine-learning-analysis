
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "dermascan_finetuned.keras"
    )

class_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

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

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("Analyze Image"):

        with st.spinner("Analyzing..."):

            model = load_model()

            img = image.resize((224, 224))
            img_array = np.array(img)

            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_array
            )

            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)

            predicted_index = np.argmax(predictions[0])

            labels = list(class_names.keys())

            predicted_label = labels[predicted_index]

            confidence = predictions[0][predicted_index] * 100

        st.success("Analysis Completed!")

        st.subheader("Prediction")

        st.write(class_names[predicted_label])

        st.write(
            f"Model prediction score: {confidence:.2f}%"
        )

        st.warning(
            "This result is experimental and is not a medical diagnosis. "
            "Please consult a qualified healthcare professional."
        )

st.markdown("---")

st.caption("DermaScan AI | Educational Project")
