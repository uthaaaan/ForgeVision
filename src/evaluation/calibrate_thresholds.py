
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.autoencoder import ConvAutoencoder
from src.models.padim import ResNetFeatureExtractor
from src.models.patchcore import PatchCoreFeatureExtractor


# Paths
DATA_DIR = ROOT / "data" / "processed" / "bottle" / "train" / "good"
SPLIT_PATH = ROOT / "data" / "processed" / "bottle" / "calibration_split.csv"

AE_PATH = ROOT / "models" / "autoencoder_best.pth"
PADIM_PATH = ROOT / "models" / "padim_parameters.pth"
PATCHCORE_PATH = ROOT / "models" / "patchcore_memory_bank.pth"

OUTPUT_JSON = ROOT / "models" / "calibration_thresholds.json"
OUTPUT_CSV = ROOT / "reports" / "calibration_scores.csv"

IMAGE_SIZE = 256
PERCENTILE = 95

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

basic_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

imagenet_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_calibration_images():

    filenames = []

    with open(SPLIT_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] == "calibration":
                filenames.append(row["filename"])

    paths = [DATA_DIR / name for name in filenames]

    missing = [path for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} calibration images are missing."
        )

    return paths


def load_models():

    # Autoencoder
    ae_checkpoint = torch.load(
        AE_PATH, map_location=DEVICE, weights_only=False
    )

    autoencoder = ConvAutoencoder().to(DEVICE)
    autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
    autoencoder.eval()

    # PaDiM
    padim_checkpoint = torch.load(
        PADIM_PATH, map_location="cpu", weights_only=False
    )

    padim_extractor = ResNetFeatureExtractor().to(DEVICE)
    padim_extractor.eval()

    selected_channels = padim_checkpoint["selected_channels"].to(DEVICE)
    padim_mean = padim_checkpoint["mean"].to(DEVICE)
    padim_inverse_covariance = (
        padim_checkpoint["inverse_covariance"].to(DEVICE)
    )

    # PatchCore
    patchcore_checkpoint = torch.load(
        PATCHCORE_PATH, map_location="cpu", weights_only=False
    )

    patchcore_extractor = PatchCoreFeatureExtractor().to(DEVICE)
    patchcore_extractor.eval()

    memory_bank = F.normalize(
        patchcore_checkpoint["memory_bank"].to(DEVICE),
        dim=1
    )

    return (
        autoencoder,
        padim_extractor,
        selected_channels,
        padim_mean,
        padim_inverse_covariance,
        patchcore_extractor,
        memory_bank
    )


def calculate_scores(image_path, models):

    (
        autoencoder,
        padim_extractor,
        selected_channels,
        padim_mean,
        padim_inverse_covariance,
        patchcore_extractor,
        memory_bank
    ) = models

    image = Image.open(image_path).convert("RGB")

    basic_image = basic_transform(image).unsqueeze(0).to(DEVICE)
    normalized_image = (
        imagenet_transform(image).unsqueeze(0).to(DEVICE)
    )

    with torch.no_grad():

        # 1. Autoencoder
        reconstruction = autoencoder(basic_image)

        error_map = torch.mean(
            (basic_image - reconstruction) ** 2,
            dim=1
        )

        ae_score = error_map.max().item()

        # 2. PaDiM
        padim_features = padim_extractor(normalized_image)

        padim_features = padim_features[:, selected_channels]

        _, channels, height, width = padim_features.shape

        padim_features = padim_features.reshape(
            1, channels, height * width
        ).permute(0, 2, 1)

        difference = padim_features - padim_mean.T.unsqueeze(0)

        squared_distance = torch.einsum(
            "bnc,ncd,bnd->bn",
            difference,
            padim_inverse_covariance,
            difference
        )

        padim_map = torch.sqrt(
            squared_distance.clamp(min=0)
        ).reshape(1, 1, height, width)

        padim_map = F.interpolate(
            padim_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        padim_score = padim_map.max().item()

        # 3. PatchCore
        patch_features = patchcore_extractor(normalized_image)

        batch, channels, height, width = patch_features.shape

        patch_features = patch_features.permute(
            0, 2, 3, 1
        ).reshape(-1, channels)

        patch_features = F.normalize(patch_features, dim=1)

        distances = torch.cdist(
            patch_features,
            memory_bank
        )

        nearest_distances = distances.min(dim=1).values

        patch_map = nearest_distances.reshape(
            batch, 1, height, width
        )

        patch_map = F.interpolate(
            patch_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        patchcore_score = patch_map.max().item()

    return {
        "autoencoder": ae_score,
        "padim": padim_score,
        "patchcore": patchcore_score
    }


def main():

    print(f"Using device: {DEVICE}")

    image_paths = load_calibration_images()

    print(f"Calibration images: {len(image_paths)}")

    print("\nLoading models...")

    models = load_models()

    print("All models loaded successfully.")

    scores = {
        "autoencoder": [],
        "padim": [],
        "patchcore": []
    }

    rows = []

    print("\nCalculating normal calibration scores...")

    for image_path in tqdm(image_paths):

        image_scores = calculate_scores(image_path, models)

        row = {"filename": image_path.name}

        for model_name, score in image_scores.items():
            scores[model_name].append(score)
            row[model_name] = score

        rows.append(row)

    # Calculate percentile thresholds
    thresholds = {}

    for model_name, model_scores in scores.items():

        thresholds[model_name] = float(
            np.percentile(model_scores, PERCENTILE)
        )

    # Save individual calibration scores
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "filename",
                "autoencoder",
                "padim",
                "patchcore"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    # Save thresholds and calibration metadata
    output = {
        "category": "bottle",
        "calibration_samples": len(image_paths),
        "percentile": PERCENTILE,
        "thresholds": thresholds
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_JSON, "w") as file:
        json.dump(output, file, indent=4)

    print("\nCALIBRATION COMPLETED")
    print("---------------------")
    print(f"Calibration samples: {len(image_paths)}")
    print(f"Threshold percentile: {PERCENTILE}")

    for model_name, threshold in thresholds.items():
        print(f"{model_name}: {threshold:.8f}")

    print(f"\nScores saved to: {OUTPUT_CSV}")
    print(f"Thresholds saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
