import os
import logging
import warnings

# Suppress all TF/Keras warnings — must happen before importing tensorflow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import json
import matplotlib.pyplot as plt

# Page config — must be first Streamlit call
st.set_page_config(
    page_title="Handwriting Digit Recognizer",
    page_icon="✏️",
    layout="centered"
)

# Load model — compile after loading to silence compile_metrics warning
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.h5")
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("✏️ Handwriting Digit Recognizer")
st.markdown("CNN trained on MNIST · 15 epochs · Data augmentation · ~99% accuracy")
st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🖊️ Draw & Predict", "📊 Training Metrics", "ℹ️ About"])

# ── Tab 1: Draw & Predict ──────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload a handwritten digit image")
    st.caption("Draw a digit (0–9) on paper, take a photo or screenshot, upload below.")

    uploaded_file = st.file_uploader(
        "Upload image (PNG, JPG, JPEG)",
        type=["png", "jpg", "jpeg"]
    )

    col1, col2 = st.columns(2)

    if uploaded_file:
        # Smart preprocessing — auto-crop to digit, match MNIST format
        def preprocess(file):
            img = Image.open(file).convert("L")

            # Auto-invert: MNIST is white digit on black — if bg is light, invert
            arr = np.array(img)
            if np.mean(arr) > 127:
                img = ImageOps.invert(img)
                arr = np.array(img)

            # Threshold to binary — removes paper texture and photo noise
            arr = (arr > 80).astype(np.uint8) * 255

            # Auto-crop: find bounding box of digit pixels
            rows = np.any(arr > 0, axis=1)
            cols = np.any(arr > 0, axis=0)
            if rows.any() and cols.any():
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                h, w = rmax - rmin, cmax - cmin
                pad = max(int(max(h, w) * 0.2), 4)
                rmin = max(0, rmin - pad)
                rmax = min(arr.shape[0], rmax + pad)
                cmin = max(0, cmin - pad)
                cmax = min(arr.shape[1], cmax + pad)
                arr = arr[rmin:rmax, cmin:cmax]

            # Make square
            h, w = arr.shape
            if h != w:
                size = max(h, w)
                square = np.zeros((size, size), dtype=np.uint8)
                square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = arr
                arr = square

            # Resize to 28x28 and normalize
            return np.array(Image.fromarray(arr).resize((28, 28), Image.Resampling.LANCZOS)) / 255.0

        img_array = preprocess(uploaded_file).reshape(1, 28, 28, 1)

        with col1:
            st.image(uploaded_file, caption="Your uploaded image", width=200)

        with col2:
            # Predict
            predictions = model.predict(img_array)
            predicted_digit = np.argmax(predictions)
            confidence = float(np.max(predictions)) * 100

            st.metric("Predicted Digit", str(predicted_digit))
            st.metric("Confidence", f"{confidence:.1f}%")

            if confidence > 85:
                st.success("High confidence prediction ✅")
            elif confidence > 60:
                st.warning("Moderate confidence — try a clearer image")
            else:
                st.error("Low confidence — model is uncertain")

        # Probability bar chart
        st.subheader("Prediction confidence across all digits")
        fig, ax = plt.subplots(figsize=(8, 3))
        bars = ax.bar(range(10), predictions[0] * 100, color=[
            "#2563eb" if i == predicted_digit else "#cbd5e1" for i in range(10)
        ])
        ax.set_xlabel("Digit")
        ax.set_ylabel("Confidence (%)")
        ax.set_xticks(range(10))
        ax.set_title(f"Model predicted: {predicted_digit} with {confidence:.1f}% confidence")
        st.pyplot(fig)

    else:
        st.info("👆 Upload an image to get started")
        st.markdown("""
        **Tips for best results:**
        - Write the digit large and centered
        - Dark pen on white paper works best
        - Good lighting, minimal shadows
        - Crop tightly around the digit before uploading
        """)

# ── Tab 2: Training Metrics ────────────────────────────────────────────────────
with tab2:
    st.subheader("Model Training Performance")

    if os.path.exists("training_history.json"):
        with open("training_history.json") as f:
            history = json.load(f)

        final_acc = history['val_accuracy'][-1] * 100
        final_loss = history['val_loss'][-1]

        m1, m2, m3 = st.columns(3)
        m1.metric("Final Val Accuracy", f"{final_acc:.2f}%")
        m2.metric("Final Val Loss", f"{final_loss:.4f}")
        m3.metric("Epochs Trained", len(history['accuracy']))

        # Accuracy plot
        fig1, ax1 = plt.subplots(figsize=(8, 3))
        ax1.plot(history['accuracy'], label='Train Accuracy', color='#2563eb')
        ax1.plot(history['val_accuracy'], label='Val Accuracy', color='#16a34a')
        ax1.set_title('Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)

        # Loss plot
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(history['loss'], label='Train Loss', color='#dc2626')
        ax2.plot(history['val_loss'], label='Val Loss', color='#ea580c')
        ax2.set_title('Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)

    else:
        st.warning("training_history.json not found. Run train_model.py first.")

# ── Tab 3: About ───────────────────────────────────────────────────────────────
with tab3:
    st.subheader("How this works")
    st.markdown("""
    **Dataset:** MNIST — 70,000 handwritten digit images (60k train, 10k test)

    **Model Architecture:**
    - Conv2D (32 filters) → MaxPooling → Dropout
    - Conv2D (64 filters) → MaxPooling → Dropout
    - Conv2D (128 filters) → Dropout
    - Dense (256) → Dropout → Dense (10, softmax)

    **Training improvements over base:**
    - ✅ Data augmentation (rotation, zoom, shift) — makes model robust to real handwriting
    - ✅ Dropout layers — prevents overfitting
    - ✅ 15 epochs instead of 5
    - ✅ Third Conv2D layer for deeper feature extraction

    **Stack:** TensorFlow · Keras · Streamlit · NumPy · Matplotlib · Pillow

    **Built by:** Garv Rana · EE Undergrad · DTU
    """)

    st.markdown("---")
    st.markdown("[GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)")
