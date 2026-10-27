
import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model_path = Path(__file__).resolve().parent / "models" / "handwritten_digit_vgg.keras"

model = tf.keras.models.load_model(model_path)

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., np.newaxis]

loss, accuracy = model.evaluate(x_test, y_test)

print(f"\nTest Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

y_pred = np.argmax(model.predict(x_test), axis=1)

print("\nClassification Report:\n")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("VGG Handwritten Digit Recognition")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.tight_layout()

output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    output_dir / "vgg_confusion_matrix.png",
    dpi=300
)

plt.show()

print(f"\nConfusion matrix saved at: {output_dir / 'vgg_confusion_matrix.png'}")
