
import tensorflow as tf
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vgg_model import build_vgg_model


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = build_vgg_model()

model.summary()

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1
)

loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model_path = Path(__file__).resolve().parent / "models" / "handwritten_digit_vgg.keras"

model_path.parent.mkdir(parents=True, exist_ok=True)

model.save(model_path)

print(f"\nVGG model saved at: {model_path}")
