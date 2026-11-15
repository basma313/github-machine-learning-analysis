
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="DermaScan AI",
    page_icon="🔬",
    layout="centered"
)

# Same class order used during training
labels = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]

class_names = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "dermascan_finetuned.keras"
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

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Analyze Image"):

            with st.spinner("Analyzing..."):

                model = load_model()

                # Check model output
                if model.output_shape[-1] != len(labels):
                    st.error("Model classes do not match.")
                    st.stop()

                # Match training preprocessing
                img_array = np.array(image)

                img_tensor = tf.convert_to_tensor(
                    img_array,
                    dtype=tf.float32
                )

                img_tensor = tf.image.resize(
                    img_tensor,
                    [224, 224]
                )

                img_tensor = tf.keras.applications.mobilenet_v2.preprocess_input(
                    img_tensor
                )

                img_tensor = tf.expand_dims(
                    img_tensor,
                    axis=0
                )

                predictions = model.predict(
                    img_tensor,
                    verbose=0
                )[0]

                scores = predictions * 100

                sorted_indices = np.argsort(scores)[::-1]

                top_index = sorted_indices[0]
                top_label = labels[top_index]
                top_score = scores[top_index]

            st.success("Analysis Completed!")

            st.subheader("Model Output")

            st.write(
                f"Highest-scoring category: "
                f"**{class_names[top_label]}**"
            )

            st.metric(
                "Top Model Score",
                f"{top_score:.2f}%"
            )

            st.caption(
                "This score is the model's output, "
                "not its actual diagnostic accuracy."
            )

            st.subheader("All Category Scores")

            results = []

            for index in sorted_indices:
                label = labels[index]

                results.append({
                    "Category": class_names[label],
                    "Model Score": f"{scores[index]:.2f}%"
                })

            st.table(results)

            st.info(
                "The model's highest score does not confirm "
                "the presence of a disease. Results are "
                "experimental and require professional "
                "medical evaluation."
            )

    except Exception as e:
        st.error(f"Error processing image: {e}")

st.markdown("---")

st.caption(
    "DermaScan AI | Educational Project"
)
