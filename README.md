# Handwriting Digit Recognition

A CNN trained on MNIST that classifies handwritten digits (0–9) from uploaded photos. Training is done with TensorFlow/Keras, and inference is deployed with ONNX Runtime in a Streamlit web app.

**Test set accuracy: 99.19%**

---

## Demo

![App Demo](app_demo.png)

---

## How it works

Upload a photo of a handwritten digit → the model preprocesses it (auto-crop, threshold, center) to match MNIST format → predicts the digit with a confidence score and full probability distribution.

---

## Model

```
Input (28×28×1)
→ Conv2D(32)  → MaxPool → Dropout(0.25)
→ Conv2D(64)  → MaxPool → Dropout(0.25)  
→ Conv2D(128)           → Dropout(0.25)
→ Dense(256)            → Dropout(0.5)
→ Dense(10, softmax)
```

Trained for 15 epochs with data augmentation (rotation, zoom, shift).

---

## Training curves

![Training Curves](training_curves.png)

| | Train | Validation |
|---|---|---|
| Accuracy | 97.05% | **99.19%** |
| Loss | 0.0967 | 0.0196 |

---

## Run locally

```bash
git clone https://github.com/garvranaaa/handwriting-digit-recognition
cd handwriting-digit-recognition

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate          # Mac/Linux

pip install -r requirements.txt
streamlit run app.py
```

The app uses the included `mnist_model.onnx` file for inference. If you want to retrain the model or regenerate weights, run:

```bash
python train_model.py
```

---

## Stack

Training: TensorFlow · Keras

Deployment: ONNX Runtime · Streamlit · NumPy · Pillow · Matplotlib

---

Garv Rana · EE Undergrad · [DTU](https://dtu.ac.in) · [GitHub](https://github.com/garvranaaa) · [LinkedIn](https://linkedin.com/in/garvsanjeevrana)
