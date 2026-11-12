
import csv
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

NORMAL_DIR = (
    ROOT / "data" / "processed" / "bottle" / "train" / "good"
)

OUTPUT_PATH = (
    ROOT / "data" / "processed" / "bottle"
    / "calibration_split.csv"
)

SEED = 42
CALIBRATION_RATIO = 0.20

images = sorted(NORMAL_DIR.glob("*.png"))

if not images:
    raise FileNotFoundError(f"No images found in {NORMAL_DIR}")

generator = torch.Generator().manual_seed(SEED)

calibration_count = round(
    len(images) * CALIBRATION_RATIO
)

training_count = len(images) - calibration_count

training_subset, calibration_subset = torch.utils.data.random_split(
    images,
    [training_count, calibration_count],
    generator=generator
)

with open(OUTPUT_PATH, "w", newline="") as file:

    writer = csv.writer(file)
    writer.writerow(["filename", "split"])

    for index in training_subset.indices:
        writer.writerow([images[index].name, "train"])

    for index in calibration_subset.indices:
        writer.writerow([images[index].name, "calibration"])

print("CONSISTENT CALIBRATION SPLIT CREATED")
print("------------------------------------")
print(f"Total images: {len(images)}")
print(f"Training images: {len(training_subset)}")
print(f"Calibration images: {len(calibration_subset)}")
print(f"Saved to: {OUTPUT_PATH}")
