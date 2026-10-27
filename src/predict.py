
import tensorflow as tf
import numpy as np
from PIL import Image

MODEL_PATH = "src/models/handwritten_digit_cnn.keras"

# Load model once
model = tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image_path):
    image = Image.open(image_path).convert("L")
    image = image.resize((28, 28))

    image_array = np.array(image)

    # Convert black digit on white background
    # to white digit on black background
    if image_array.mean() > 127:
        image_array = 255 - image_array

    image_array = image_array.astype("float32") / 255.0

    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


def predict_digit(image_path):
    image = preprocess_image(image_path)

    probabilities = model.predict(image, verbose=0)[0]

    digit = int(np.argmax(probabilities))
    confidence = float(probabilities[digit]) * 100

    print("\n========== DIGIT RECOGNITION ==========")
    print("Image:", image_path)
    print("Predicted Digit:", digit)
    print(f"Confidence: {confidence:.2f}%")
    print("=====================================")

    return digit, confidence


if __name__ == "__main__":
    image_path = input("Enter image path: ")
    predict_digit(image_path)
