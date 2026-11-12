
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.padim import ResNetFeatureExtractor


# Configuration
DATA_DIR = ROOT / "data" / "processed" / "bottle" / "train" / "good"
SPLIT_PATH = ROOT / "data" / "processed" / "bottle" / "calibration_split.csv"
OUTPUT_PATH = ROOT / "models" / "padim_parameters.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8
FEATURE_DIM = 100
SEED = 42
REGULARIZATION = 0.01

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class NormalImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


def main():

    print(f"Using device: {DEVICE}")

    # Read the consistent training/calibration split
    training_filenames = []

    with open(SPLIT_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["split"] == "train":
                training_filenames.append(row["filename"])

    image_paths = [
        DATA_DIR / filename
        for filename in training_filenames
    ]

    missing = [path for path in image_paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training images are missing."
        )

    print(f"Training images: {len(image_paths)}")

    # ImageNet normalization
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = NormalImageDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda")
    )

    # Feature extractor
    torch.manual_seed(SEED)

    feature_extractor = ResNetFeatureExtractor().to(DEVICE)
    feature_extractor.eval()

    # Select the same 100 feature channels deterministically
    selected_channels = torch.randperm(448)[:FEATURE_DIM]

    extracted_features = []

    print("\nExtracting normal image features...")

    with torch.no_grad():

        for images in tqdm(loader):

            images = images.to(DEVICE)

            features = feature_extractor(images)

            features = features[:, selected_channels.to(DEVICE)]

            extracted_features.append(
                features.cpu()
            )

    # Shape: [167, 100, 64, 64]
    features = torch.cat(extracted_features, dim=0)

    print(f"\nExtracted feature shape: {features.shape}")

    num_samples, feature_dim, height, width = features.shape
    num_locations = height * width

    # Rearrange to [samples, spatial locations, channels]
    features = features.reshape(
        num_samples, feature_dim, num_locations
    ).permute(0, 2, 1).contiguous()

    # Calculate Gaussian distribution at each spatial location
    print("\nCalculating mean and covariance...")

    mean = features.mean(dim=0)

    centered = features - mean.unsqueeze(0)

    identity = torch.eye(FEATURE_DIM, device=DEVICE)

    inverse_covariance = torch.empty(
        num_locations,
        FEATURE_DIM,
        FEATURE_DIM,
        dtype=torch.float32
    )

    # Process spatial locations in chunks to limit GPU memory use
    chunk_size = 256

    for start in tqdm(range(0, num_locations, chunk_size)):

        end = min(start + chunk_size, num_locations)

        chunk = centered[:, start:end, :].to(DEVICE)

        covariance = torch.einsum(
            "nsc,nsd->scd",
            chunk,
            chunk
        ) / (num_samples - 1)

        covariance += REGULARIZATION * identity.unsqueeze(0)

        inverse = torch.linalg.inv(covariance)

        inverse_covariance[start:end] = inverse.cpu()

    # Save model parameters
    checkpoint = {
        "mean": mean.T.contiguous(),
        "inverse_covariance": inverse_covariance,
        "selected_channels": selected_channels,
        "feature_dim": FEATURE_DIM,
        "feature_height": height,
        "feature_width": width,
        "training_samples": num_samples,
        "regularization": REGULARIZATION
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, OUTPUT_PATH)

    print("\nPADIM TRAINING COMPLETED")
    print("------------------------")
    print(f"Training samples: {num_samples}")
    print(f"Selected channels: {FEATURE_DIM}")
    print(f"Spatial locations: {num_locations}")
    print(f"Mean shape: {checkpoint['mean'].shape}")
    print(
        "Inverse covariance shape:",
        checkpoint["inverse_covariance"].shape
    )
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
