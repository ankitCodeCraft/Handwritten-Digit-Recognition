
import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test[..., np.newaxis]

models_dir = Path(__file__).resolve().parent / "models"

model_list = [
    ("CNN", models_dir / "handwritten_digit_cnn.keras"),
    ("ResNet", models_dir / "handwritten_digit_resnet.keras"),
    ("VGG", models_dir / "handwritten_digit_vgg.keras"),
    ("EfficientNetB0", models_dir / "handwritten_digit_efficientnet.keras")
]

results = []

for model_name, model_path in model_list:
    print(f"\nEvaluating {model_name}...")

    model = tf.keras.models.load_model(model_path)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    predictions = np.argmax(
        model.predict(x_test, verbose=0), axis=1
    )

    precision = precision_score(
        y_test, predictions, average="macro", zero_division=0
    )

    recall = recall_score(
        y_test, predictions, average="macro", zero_division=0
    )

    f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy (%)": round(accuracy * 100, 2),
        "Loss": round(loss, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "Parameters": model.count_params()
    })

comparison = pd.DataFrame(results)

print("\nMODEL COMPARISON\n")
print(comparison.to_string(index=False))

comparison_path = Path(__file__).resolve().parent / "model_comparison.csv"
comparison.to_csv(comparison_path, index=False)

print(f"\nComparison saved at: {comparison_path}")
