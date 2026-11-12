
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.padim import ResNetFeatureExtractor


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

DATA_DIR = ROOT / "data" / "processed" / "bottle"
PARAMS_PATH = ROOT / "models" / "padim_parameters.pth"

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load model and PaDiM parameters
model = ResNetFeatureExtractor().to(DEVICE)
model.eval()

params = torch.load(
    PARAMS_PATH,
    map_location="cpu",
    weights_only=True
)

means = params["mean"].to(DEVICE)
inverse_covariance = params["inverse_covariance"].to(DEVICE)
selected_channels = params["selected_channels"].to(DEVICE)

H = params["feature_height"]
W = params["feature_width"]


def get_anomaly_map(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        features = model(tensor)
        features = features[:, selected_channels]

        # [D, H*W]
        features = features.squeeze(0).reshape(
            len(selected_channels), -1
        )

        # Difference from learned normal distribution
        difference = features.T - means.T

        # Mahalanobis distance
        distance = torch.einsum(
            "nd,ndk,nk->n",
            difference,
            inverse_covariance,
            difference
        )

        anomaly_map = distance.reshape(1, 1, H, W)

        anomaly_map = F.interpolate(
            anomaly_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

    return anomaly_map.squeeze().cpu().numpy()


# Test dataset
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

normal_scores = []
defective_scores = []
inference_times = []

print("\nPADIM EVALUATION")
print("----------------")
print(f"Device: {DEVICE}")
print(f"Normal images: {len(normal_images)}")
print(f"Defective images: {len(defective_images)}")


# Evaluate normal images
for image_path in normal_images:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    anomaly_map = get_anomaly_map(image_path)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_times.append(time.perf_counter() - start)

    score = anomaly_map.max()
    normal_scores.append(score)

    image_labels.append(0)
    image_scores.append(score)

    pixel_labels.extend([0] * anomaly_map.size)
    pixel_scores.extend(anomaly_map.ravel())


# Evaluate defective images
for image_path in defective_images:
    mask_path = mask_dir / f"{image_path.stem}.png"

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    anomaly_map = get_anomaly_map(image_path)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    inference_times.append(time.perf_counter() - start)

    mask = Image.open(mask_path).convert("L").resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask_array = np.array(mask) > 0

    score = anomaly_map.max()
    defective_scores.append(score)

    image_labels.append(1)
    image_scores.append(score)

    pixel_labels.extend(mask_array.ravel().astype(np.uint8))
    pixel_scores.extend(anomaly_map.ravel())


# Metrics
image_auc = roc_auc_score(image_labels, image_scores)

pixel_auc = roc_auc_score(
    np.asarray(pixel_labels),
    np.asarray(pixel_scores)
)

print("\nEVALUATION RESULTS")
print("------------------")
print(f"Image-level AUROC: {image_auc:.4f}")
print(f"Pixel-level AUROC: {pixel_auc:.4f}")

print("\nANOMALY SCORES")
print("--------------")
print(f"Average normal score: {np.mean(normal_scores):.6f}")
print(f"Average defective score: {np.mean(defective_scores):.6f}")

print("\nINFERENCE PERFORMANCE")
print("---------------------")
print(f"Average inference time: {np.mean(inference_times) * 1000:.2f} ms")
print(f"Total evaluated images: {len(inference_times)}")

print("\nPADIM EVALUATION COMPLETE")
