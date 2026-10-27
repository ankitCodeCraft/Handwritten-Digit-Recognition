
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_dir = Path(__file__).resolve().parent
df = pd.read_csv(project_dir / "model_comparison.csv")

output_dir = project_dir / "results"
output_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 5))

bars = plt.bar(df["Model"], df["Accuracy (%)"])

plt.title("CNN, ResNet, VGG and EfficientNetB0 Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 105)

for bar, value in zip(bars, df["Accuracy (%)"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
accuracy_path = output_dir / "accuracy_comparison.png"
plt.savefig(accuracy_path, dpi=300)
plt.show()

print(f"Accuracy graph saved at: {accuracy_path}")

plt.figure(figsize=(9, 5))

bars = plt.bar(df["Model"], df["F1 Score"])

plt.title("CNN, ResNet, VGG and EfficientNetB0 F1 Score Comparison")
plt.xlabel("Model")
plt.ylabel("Macro F1 Score")
plt.ylim(0, 1.1)

for bar, value in zip(bars, df["F1 Score"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.02,
        f"{value:.4f}",
        ha="center"
    )

plt.tight_layout()
f1_path = output_dir / "f1_comparison.png"
plt.savefig(f1_path, dpi=300)
plt.show()

print(f"F1 graph saved at: {f1_path}")
