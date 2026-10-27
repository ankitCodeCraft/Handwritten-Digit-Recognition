
import tensorflow as tf
from pathlib import Path
from efficientnet_model import build_efficientnet_model

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = build_efficientnet_model()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=2,
        restore_best_weights=True
    )
]

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    callbacks=callbacks
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

model_path = Path(__file__).resolve().parent / "models" / "handwritten_digit_efficientnet.keras"
model_path.parent.mkdir(parents=True, exist_ok=True)
model.save(model_path)

print(f"\nModel saved at: {model_path}")
