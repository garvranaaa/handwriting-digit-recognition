# Handwritten Digit Recognition

A Streamlit-based MNIST digit recognition app with separate training and deployment files.

## Project structure

- `train_model.py` — trains the MNIST model with data augmentation, dropout, and 15 epochs, then saves:
  - `mnist_model.h5`
  - `training_history.json`
  - `training_curves.png`
- `app.py` — Streamlit app that loads the saved model and predicts handwritten digit uploads.
- `requirements.txt` — dependencies for training and running the app.

## Features

- 15-epoch convolutional model with data augmentation
- Dropout layers to reduce overfitting
- Saves a pre-trained model for instant Streamlit serving
- Upload any handwritten digit image and see:
  - predicted digit
  - confidence bar chart
  - top prediction probabilities
- Training accuracy/loss curves shown in the UI

## Run instructions

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Output files

- `mnist_model.h5`: saved trained model
- `training_history.json`: epoch-by-epoch metrics
- `training_curves.png`: training curves image

## Demo

![Streamlit App Demo](app_demo.png)

### Training curves

![Training Curves](training_curves.png)

## Result

The upgraded training workflow achieved a test accuracy of **99.19%**.

## Notes

- The app uses image upload rather than a canvas widget for better compatibility across deployment targets.
- If you want to update the demo screenshot later, replace `app_demo.png` with a real Streamlit screenshot.
