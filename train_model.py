import json
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dropout, Flatten, Dense, MaxPooling2D, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_model() -> tf.keras.Model:
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    train_datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
    )

    batch_size = 64
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)

    model = build_model()
    history = model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        epochs=15,
        validation_data=(X_test, y_test),
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    model.save("mnist_model.h5")

    history_data = {
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "val_accuracy": [float(x) for x in history.history["val_accuracy"]],
        "loss": [float(x) for x in history.history["loss"]],
        "val_loss": [float(x) for x in history.history["val_loss"]],
    }

    with open("training_history.json", "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(history_data["accuracy"], label="Train Accuracy", marker="o")
    ax[0].plot(history_data["val_accuracy"], label="Validation Accuracy", marker="o")
    ax[0].set_title("Training Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(history_data["loss"], label="Train Loss", marker="o")
    ax[1].plot(history_data["val_loss"], label="Validation Loss", marker="o")
    ax[1].set_title("Training Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    fig.tight_layout()
    fig.savefig("training_curves.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
