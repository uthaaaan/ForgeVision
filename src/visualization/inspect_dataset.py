
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT_ROOT / "data/processed/bottle/manifest.csv"
OUTPUT = PROJECT_ROOT / "reports/bottle_dataset_preview.png"


def load_samples():
    with open(MANIFEST, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main():
    samples = load_samples()

    normal = next(
        s for s in samples
        if s["split"] == "train" and s["defect"] == "good"
    )

    defective = next(
        s for s in samples
        if s["defect"] == "broken_small"
    )

    normal_image = np.array(
        Image.open(
            PROJECT_ROOT / normal["processed_image"]
        ).convert("RGB")
    )

    defect_image = np.array(
        Image.open(
            PROJECT_ROOT / defective["processed_image"]
        ).convert("RGB")
    )

    mask = np.array(
        Image.open(
            PROJECT_ROOT / defective["processed_mask"]
        ).convert("L")
    )

    overlay = defect_image.copy()
    defect_pixels = mask > 0

    red = np.array([255, 0, 0], dtype=np.uint8)

    overlay[defect_pixels] = (
        0.5 * overlay[defect_pixels] + 0.5 * red
    ).astype(np.uint8)

    fig, axes = plt.subplots(
        1, 4,
        figsize=(14, 4)
    )

    images = [
        normal_image,
        defect_image,
        mask,
        overlay
    ]

    titles = [
        "Normal Bottle",
        "Defective Bottle",
        "Ground Truth Mask",
        "Defect Overlay"
    ]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray" if title == "Ground Truth Mask" else None)
        ax.set_title(title, fontsize=11, pad=10)
        ax.axis("off")

    fig.suptitle(
        "ForgeVision - Dataset Inspection",
        fontsize=14,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Saved visualization: {OUTPUT}")


if __name__ == "__main__":
    main()
