
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.autoencoder import ConvAutoencoder


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

DATA_DIR = ROOT / "data" / "processed" / "bottle"
CHECKPOINT = ROOT / "models" / "autoencoder_best.pth"
OUTPUT_DIR = ROOT / "reports" / "reconstruction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])


# Load model
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=True
)

model = ConvAutoencoder().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def inspect_image(image_path, mask_path=None):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        reconstructed = model(tensor)

    original = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    reconstruction = reconstructed.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    error_map = np.mean(
        (original - reconstruction) ** 2,
        axis=2
    )

    print(f"\nImage: {image_path.name}")
    print(f"Mean reconstruction error: {error_map.mean():.8f}")
    print(f"Maximum pixel error: {error_map.max():.8f}")
    print(f"Minimum pixel error: {error_map.min():.8f}")

    if mask_path:
        mask = Image.open(mask_path).convert("L").resize(
            (IMAGE_SIZE, IMAGE_SIZE)
        )
        mask_array = np.array(mask) > 0
        defect_error = error_map[mask_array]
        normal_error = error_map[~mask_array]

        print(f"Ground-truth defect pixels: {mask_array.sum()}")
        print(
            f"Mean error on defect pixels: "
            f"{defect_error.mean():.8f}"
        )
        print(
            f"Mean error on normal pixels: "
            f"{normal_error.mean():.8f}"
        )

    # Save visualization
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    axes[1].imshow(reconstruction)
    axes[1].set_title("Reconstruction")

    axes[2].imshow(error_map, cmap="inferno")
    axes[2].set_title("Reconstruction Error")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{image_path.stem}_comparison.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Visualization saved: {output_path}")


# Select one normal and one defective image
normal_images = sorted(
    (DATA_DIR / "test" / "good").glob("*.png")
)

defective_images = sorted(
    (DATA_DIR / "test" / "defective" / "images").glob("*.png")
)

mask_dir = DATA_DIR / "test" / "defective" / "masks"

if not normal_images or not defective_images:
    raise RuntimeError("Test images not found.")

print("AUTOENCODER RECONSTRUCTION INSPECTION")
print("-------------------------------------")
print(f"Device: {DEVICE}")
print(f"Checkpoint epoch: {checkpoint['epoch']}")
print(f"Best validation loss: {checkpoint['best_val_loss']:.8f}")

inspect_image(normal_images[0])

defective_image = defective_images[0]
mask_path = mask_dir / f"{defective_image.stem}.png"

if not mask_path.exists():
    raise FileNotFoundError(f"Mask not found: {mask_path}")

inspect_image(defective_image, mask_path)

print("\nINSPECTION COMPLETE")
