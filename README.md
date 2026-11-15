# DermaScan AI

## About the Project
DermaScan AI is a machine learning project that I built to explore how AI can help with skin image classification.

The idea started because I wanted to use AI in a way that could be useful in healthcare. The project allows users to upload a skin image and get a prediction of its possible category.

## The Problem
Skin conditions can look very similar to each other, which can make identifying them difficult. I wanted to explore whether a machine learning model could learn to recognize different skin lesion patterns from images.

## My Solution
I built a web application using Python, TensorFlow, and Streamlit.

The model analyzes an uploaded image and predicts which of the seven categories it may belong to:

- Actinic Keratoses (akiec)
- Basal Cell Carcinoma (bcc)
- Benign Keratosis (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic Nevi (nv)
- Vascular Lesions (vasc)

The application also displays the prediction scores for all seven categories.

## Dataset
I used the HAM10000 dataset, which contains more than 10,000 dermatoscopic images of different skin lesions.

The images
