import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import pickle
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("cnn_model.h5")


# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to the required dimensions
    img = cv2.resize(np.array(image), (128, 128))
    img = img / 255.0  # Normalize the image
    img = np.reshape(img, (1, 128, 128, 3))  # Reshape for model input
    return img

# Prediction function
def predict_image(img):
    pred = model.predict(img)
    return 1 if pred[0]>=0.5 else 0

# Streamlit App UI
st.title("College Uniform Classification")
st.write("Upload an image or take a photo to check if the person is wearing a college uniform.")

# Image input section
image_input = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Capture an Image")

if image_input or camera_input:
    # Load the image
    if image_input:
        image = Image.open(image_input)
    else:
        image = Image.open(camera_input)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image and predict
    processed_image = preprocess_image(image)
    prediction = predict_image(processed_image)

    # Display the prediction
    st.write("### Prediction:")
    if prediction==1:
        st.success('wearing uniform')
    else:
        st.warning('not wearing uniform')

# Style enhancements
st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
        background-color: #f7f9fc;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)
