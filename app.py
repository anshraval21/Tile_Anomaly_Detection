import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# Disable scientific notation
np.set_printoptions(suppress=True)

# Define paths
MODEL_PATH = "saved_model/model.savedmodel"
LABEL_PATH = "saved_model/labels.txt"

# Load model
if not os.path.isdir(MODEL_PATH):
    st.error(f"❌ Model directory not found at: {MODEL_PATH}")
    st.stop()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Load class labels
try:
    with open(LABEL_PATH, "r") as f:
        class_names = [line.strip().split(" ", 1)[-1] for line in f.readlines()]
except Exception as e:
    st.error(f"❌ Failed to load labels: {e}")
    st.stop()

# Check label count
if len(class_names) != 2:
    st.error("❌ Expected exactly 2 labels in labels.txt (e.g., 'Good' and 'Defect').")
    st.stop()

# Streamlit UI
st.title("🧠 Tile Defect Finder")
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess image
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    image_array = np.asarray(image).astype(np.float32)
    normalized_image_array = (image_array / 127.5) - 1
    data = np.expand_dims(normalized_image_array, axis=0)

    # Predict
    prediction = model.predict(data)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = float(prediction[predicted_index])

    # Show result
    if predicted_class.lower() == "good" and confidence >= 0.9945:
        st.success("✅ Product is GOOD (100%)")
    else:
        st.error(f"❌ Product is DEFECTED ({confidence * 100:.2f}%)")

    # Show prediction summary
    st.markdown(f"### 🔍 Top Prediction: `{predicted_class}` ({confidence * 100:.2f}%)")

    # Class probability bar
    st.subheader("📊 Class Probabilities")
    for label, score in zip(class_names, prediction):
        score = float(score)
        st.progress(score, text=f"{label} — {score * 100:.2f}%")
