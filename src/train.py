
import os
import sys
import tensorflow as tf
import matplotlib.pyplot as plt

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.models.cnn_model import build_cnn_model


def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.
    """

    (x_train, y_train), (x_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension for CNN
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    print("Training images:", x_train.shape)
    print("Training labels:", y_train.shape)
    print("Testing images:", x_test.shape)
    print("Testing labels:", y_test.shape)

    return x_train, y_train, x_test, y_test


def train_model():
    """
    Train and evaluate the CNN model.
    """

    x_train, y_train, x_test, y_test = load_mnist_data()

    model = build_cnn_model()

    model.summary()

    # Create model output directory
    model_dir = os.path.join(PROJECT_ROOT, "src", "models")
    os.makedirs(model_dir, exist_ok=True)

    # Train the model
    history = model.fit(
        x_train,
        y_train,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(
        x_test,
        y_test,
        verbose=1
    )

    print("\nTest Loss:", test_loss)
    print("Test Accuracy:", test_accuracy)

    # Save trained model
    model_path = os.path.join(model_dir, "handwritten_digit_cnn.keras")
    model.save(model_path)

    print("\nModel saved successfully:", model_path)

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("CNN Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("CNN Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    train_model()
