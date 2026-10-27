import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Load MNIST test dataset
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize images
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_test = x_test.reshape(-1, 28, 28, 1)

# Load trained model
model = tf.keras.models.load_model(
    "src/models/handwritten_digit_cnn.keras"
)

# Predict digits
predictions = model.predict(x_test)

# Convert probabilities to predicted labels
y_pred = np.argmax(predictions, axis=1)

# Classification report
print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=range(10)
)

disp.plot(cmap="Blues")

plt.title("CNN Handwritten Digit Recognition")
plt.show()