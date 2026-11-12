from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = ROOT / "reports" / "baseline_results.csv"
OUTPUT_PATH = ROOT / "reports" / "model_comparison.png"

# Load evaluation results
df = pd.read_csv(RESULTS_PATH)

models = df["Model"].tolist()

image_auroc = df["Image AUROC"].tolist()
pixel_auroc = df["Pixel AUROC"].tolist()
inference_time = df["Average Inference (ms)"].tolist()

# Create comparison figure
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

x = np.arange(len(models))
width = 0.55

# Image-level AUROC
axes[0].bar(x, image_auroc, width)
axes[0].set_title("Image-Level AUROC")
axes[0].set_ylabel("AUROC")
axes[0].set_ylim(0, 1.1)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models, rotation=20, ha="right")
axes[0].grid(axis="y", alpha=0.3)

# Pixel-level AUROC
axes[1].bar(x, pixel_auroc, width)
axes[1].set_title("Pixel-Level AUROC")
axes[1].set_ylabel("AUROC")
axes[1].set_ylim(0, 1.1)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models, rotation=20, ha="right")
axes[1].grid(axis="y", alpha=0.3)

# Inference time
axes[2].bar(x, inference_time, width)
axes[2].set_title("Average Inference Time")
axes[2].set_ylabel("Milliseconds")
axes[2].set_xticks(x)
axes[2].set_xticklabels(models, rotation=20, ha="right")
axes[2].grid(axis="y", alpha=0.3)

# Add value labels
for ax in axes:
    for bar in ax.patches:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}" if ax != axes[2] else f"{height:.2f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=9,
            xytext=(0, 4),
            textcoords="offset points"
        )

fig.suptitle(
    "ForgeVision: Industrial Anomaly Detection Model Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("MODEL COMPARISON VISUALIZATION COMPLETE")
print(f"Saved to: {OUTPUT_PATH}")