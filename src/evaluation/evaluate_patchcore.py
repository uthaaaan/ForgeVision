
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.patchcore import PatchCoreFeatureExtractor


# Paths
DATA_ROOT = ROOT / "data" / "processed" / "bottle"
CHECKPOINT = ROOT / "models" / "patchcore_memory_bank.pth"

NORMAL_DIR = DATA_ROOT / "test" / "good"
DEFECT_DIR = DATA_ROOT / "test" / "defective" / "images"
MASK_DIR = DATA_ROOT / "test" / "defective" / "masks"

IMAGE_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_image_paths(directory):
    extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    paths = []

    for extension in extensions:
        paths.extend(directory.glob(extension))

    return sorted(paths)


def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(DEVICE)


def load_mask(path):
    mask = Image.open(path).convert("L")
    mask = mask.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    mask = np.array(mask)
    return (mask > 0).astype(np.uint8)


def calculate_anomaly_map(features, memory_bank):
    # Convert feature map into normalized patch vectors.
    patches = features.permute(0, 2, 3, 1).reshape(-1, 384)
    patches = F.normalize(patches.float(), dim=1)

    # Find each patch's nearest memory-bank distance.
    distance_chunks = []

    for start in range(0, len(patches), 256):
        query = patches[start:start + 256]

        distances = torch.cdist(
            query,
            memory_bank
        )

        nearest_distances = distances.min(dim=1).values
        distance_chunks.append(nearest_distances)

    distances = torch.cat(distance_chunks)

    anomaly_map = distances.reshape(
        1, 1, features.shape[2], features.shape[3]
    )

    anomaly_map = F.interpolate(
        anomaly_map,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    return anomaly_map.squeeze().cpu().numpy()


def main():
    print(f"Device: {DEVICE}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True
    )

    memory_bank = checkpoint["memory_bank"].to(DEVICE)
    memory_bank = F.normalize(memory_bank.float(), dim=1)

    model = PatchCoreFeatureExtractor().to(DEVICE)
    model.eval()

    normal_paths = get_image_paths(NORMAL_DIR)
    defect_paths = get_image_paths(DEFECT_DIR)

    if not normal_paths or not defect_paths:
        raise RuntimeError("Test images were not found.")

    # Match masks to defective images by filename.
    mask_paths = {
        path.stem: path for path in get_image_paths(MASK_DIR)
    }

    if any(path.stem not in mask_paths for path in defect_paths):
        raise RuntimeError("A defective image is missing its matching mask.")

    image_scores = []
    image_labels = []
    pixel_scores = []
    pixel_labels = []
    inference_times = []

    print(f"Normal test images: {len(normal_paths)}")
    print(f"Defective test images: {len(defect_paths)}")
    print("\nEvaluating PatchCore...")

    test_items = (
        [(path, 0, None) for path in normal_paths]
        + [
            (path, 1, mask_paths[path.stem])
            for path in defect_paths
        ]
    )

    with torch.no_grad():
        for image_path, label, mask_path in tqdm(test_items):
            image = load_image(image_path)

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            start_time = time.perf_counter()

            features = model(image)
            anomaly_map = calculate_anomaly_map(
                features,
                memory_bank
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            inference_times.append(
                (time.perf_counter() - start_time) * 1000
            )

            # Maximum anomaly score for image-level detection.
            image_scores.append(float(anomaly_map.max()))
            image_labels.append(label)

            if label == 0:
                ground_truth = np.zeros(
                    (IMAGE_SIZE, IMAGE_SIZE),
                    dtype=np.uint8
                )
            else:
                ground_truth = load_mask(mask_path)

            pixel_scores.extend(anomaly_map.ravel())
            pixel_labels.extend(ground_truth.ravel())

    image_auc = roc_auc_score(image_labels, image_scores)
    pixel_auc = roc_auc_score(pixel_labels, pixel_scores)
    average_time = np.mean(inference_times)

    normal_scores = [
        score for score, label in zip(image_scores, image_labels)
        if label == 0
    ]

    defect_scores = [
        score for score, label in zip(image_scores, image_labels)
        if label == 1
    ]

    print("\nPATCHCORE EVALUATION RESULTS")
    print("----------------------------")
    print(f"Image-level AUROC: {image_auc:.4f}")
    print(f"Pixel-level AUROC: {pixel_auc:.4f}")
    print(f"Average normal anomaly score: {np.mean(normal_scores):.6f}")
    print(f"Average defective anomaly score: {np.mean(defect_scores):.6f}")
    print(f"Average inference time: {average_time:.2f} ms")
    print(f"Total evaluated images: {len(test_items)}")


if __name__ == "__main__":
    main()
