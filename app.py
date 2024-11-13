#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 22:22:22 2024

@author: seynoma
"""
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Set up Streamlit page configuration
st.set_page_config(page_title="Image Classifier", page_icon="🖼️", layout="wide")

# Load your trained model
MODEL_PATH = "runs/classify/train/weights/best.pt"
model = YOLO(MODEL_PATH)

# Streamlit app interface
st.title("🖼️ Image Classifier with YOLOv8")
st.markdown("Upload an image and let the model classify it into categories!")

# Image uploader
uploaded_image = st.file_uploader("Choose an image to upload:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Open and preprocess the image
    image = Image.open(uploaded_image).resize((416, 416))

    # Perform inference
    results = model.predict(image)

    # Display predictions
    st.subheader("Prediction Results:")
    for result in results:
        predicted_class = result.names[result.probs.argmax()]
        confidence = result.probs.max().item()
        st.write(f"**Category:** {predicted_class} - **Confidence:** {confidence:.2f}")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using [YOLOv8](https://github.com/ultralytics/ultralytics) and [Streamlit](https://streamlit.io).")

