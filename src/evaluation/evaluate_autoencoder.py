
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.autoencoder import ConvAutoencoder


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

DATA_DIR = ROOT / "data" / "processed" / "bottle"
CHECKPOINT = ROOT / "models" / "autoencoder_best.pth"

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


def get_error_map(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        reconstructed = model(tensor)

    original = tensor.squeeze(0)
    reconstructed = reconstructed.squeeze(0)

    error_map = torch.mean(
        (original - reconstructed) ** 2,
        dim=0
    )

    return error_map.cpu().numpy()


# Collect test images
normal_images = sorted(
    (DATA_DIR / "test" / "good").glob("*.png")
)

defective_images = sorted(
    (DATA_DIR / "test" / "defective" / "images").glob("*.png")
)

mask_dir = DATA_DIR / "test" / "defective" / "masks"

image_labels = []
image_scores = []

pixel_labels = []
pixel_scores = []

normal_errors = []
defective_errors = []

inference_times = []

print("\nAUTOENCODER EVALUATION")
print("----------------------")
print(f"Device: {DEVICE}")
print(f"Normal test images: {len(normal_images)}")
print(f"Defective test images: {len(defective_images)}")


# Evaluate normal images
for image_path in normal_images:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    error_map = get_error_map(image_path)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_times.append(time.perf_counter() - start)

    normal_errors.append(error_map.mean())

    image_labels.append(0)
    image_scores.append(error_map.max())

    pixel_labels.extend([0] * error_map.size)
    pixel_scores.extend(error_map.ravel())


# Evaluate defective images
for image_path in defective_images:
    mask_path = mask_dir / f"{image_path.stem}.png"

    if not mask_path.exists():
        raise FileNotFoundError(f"Missing mask: {mask_path}")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    error_map = get_error_map(image_path)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_times.append(time.perf_counter() - start)

    mask = Image.open(mask_path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask_array = np.array(mask) > 0

    if mask_array.shape != error_map.shape:
        raise ValueError(f"Mask size mismatch: {image_path.name}")

    defective_errors.append(error_map.mean())

    image_labels.append(1)
    image_scores.append(error_map.max())

    pixel_labels.extend(mask_array.ravel().astype(np.uint8))
    pixel_scores.extend(error_map.ravel())


# Calculate metrics
image_auc = roc_auc_score(image_labels, image_scores)

pixel_auc = roc_auc_score(
    np.asarray(pixel_labels),
    np.asarray(pixel_scores)
)

print("\nEVALUATION RESULTS")
print("------------------")
print(f"Image-level AUROC: {image_auc:.4f}")
print(f"Pixel-level AUROC: {pixel_auc:.4f}")

print("\nRECONSTRUCTION ERROR")
print("--------------------")
print(f"Average normal image error: {np.mean(normal_errors):.8f}")
print(f"Average defective image error: {np.mean(defective_errors):.8f}")

print("\nINFERENCE PERFORMANCE")
print("----------------------")
print(f"Average inference time: {np.mean(inference_times) * 1000:.2f} ms")
print(f"Total evaluated images: {len(inference_times)}")

print("\nEVALUATION COMPLETE")
