import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Load the trained model
model = tf.keras.models.load_model("DenseNet_noTop_MRI.keras")

# Define class names
class_names = ["Lung ACA", "Lung N", "Lung SCC", "Colon ACA", "Colon N"]

# Function to preprocess uploaded image
def preprocess_image(image, target_size=(200, 200)):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image) / 255.0  # Normalize the image
    return np.expand_dims(image, axis=0)  # Add batch dimension

# Function to predict and display results
def predict_image(model, image, class_names):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class, predictions[0]

# Streamlit app layout
st.title("Lung and Colon Cancer Image Classification")
st.sidebar.title("Options")
app_mode = st.sidebar.selectbox("Choose the mode", ["Upload Image", "View Metrics"])

# Sidebar to select app mode
if app_mode == "Upload Image":
    st.header("Upload an Image for Classification")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        st.write("Classifying...")
        predicted_class, predictions = predict_image(model, image, class_names)
        
        st.write(f"Prediction: **{predicted_class}**")
        st.write("Prediction Confidence:")
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {predictions[i]:.2f}")
        st.bar_chart(predictions)

elif app_mode == "View Metrics":
    st.header("Model Training Metrics")
    
    # Placeholder data for training metrics - replace with actual training history data if available
    acc = [0.8, 0.85, 0.9]  # Example data - replace with history.history['accuracy']
    val_acc = [0.78, 0.84, 0.88]  # Example data - replace with history.history['val_accuracy']
    loss = [0.5, 0.4, 0.3]  # Example data - replace with history.history['loss']
    val_loss = [0.52, 0.45, 0.38]  # Example data - replace with history.history['val_loss']
    
    # Plot accuracy and loss metrics
    st.subheader("Accuracy over Epochs")
    st.line_chart({"Train Accuracy": acc, "Validation Accuracy": val_acc})

    st.subheader("Loss over Epochs")
    st.line_chart({"Train Loss": loss, "Validation Loss": val_loss})

    # Optionally, add additional metrics visualizations or confusion matrix here
