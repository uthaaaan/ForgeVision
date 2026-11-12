
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.inference.inference_engine import ForgeVisionInference


# Load inference engine
engine = ForgeVisionInference()

# Select a defective test image
defective_dir = (
    ROOT / "data" / "processed" / "bottle"
    / "test" / "defective" / "images"
)

images = sorted(defective_dir.glob("*.png"))

if not images:
    raise FileNotFoundError("No defective test images found.")

image_path = images[0]
print(f"Testing image: {image_path.name}")

# Generate Autoencoder anomaly map
output = engine.predict(image_path)

original = output["original_image"].resize((256, 256))
raw_map = output["results"]["Autoencoder"]["anomaly_map"]

# Gaussian smoothing
smoothed_map = gaussian_filter(raw_map, sigma=2.0)

# Robust percentile normalization
lower = np.percentile(smoothed_map, 1)
upper = np.percentile(smoothed_map, 99.5)

normalized_map = np.clip(
    (smoothed_map - lower) / (upper - lower + 1e-8),
    0,
    1
)

# Generate heatmaps
raw_normalized = (
    raw_map - raw_map.min()
) / (
    raw_map.max() - raw_map.min() + 1e-8
)

raw_heatmap = plt.get_cmap("inferno")(
    raw_normalized
)[:, :, :3]

improved_heatmap = plt.get_cmap("inferno")(
    normalized_map
)[:, :, :3]

raw_overlay = (
    0.55 * np.asarray(original) / 255
    + 0.45 * raw_heatmap
)

improved_overlay = (
    0.55 * np.asarray(original) / 255
    + 0.45 * improved_heatmap
)

# Create comparison
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

axes[0].imshow(original)
axes[0].set_title("Original Image")

axes[1].imshow(raw_map, cmap="inferno")
axes[1].set_title("Raw Anomaly Map")

axes[2].imshow(raw_overlay.clip(0, 1))
axes[2].set_title("Raw Overlay")

axes[3].imshow(improved_heatmap)
axes[3].set_title("Smoothed Heatmap")

axes[4].imshow(improved_overlay.clip(0, 1))
axes[4].set_title("Improved Overlay")

for axis in axes:
    axis.axis("off")

plt.tight_layout()

output_path = ROOT / "reports" / "autoencoder_heatmap_comparison.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nCOMPARISON COMPLETED")
print(f"Saved to: {output_path}")
