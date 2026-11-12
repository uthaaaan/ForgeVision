
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

from src.models.patchcore import (
    PatchCoreFeatureExtractor,
    greedy_coreset
)


# Configuration
DATA_DIR = ROOT / "data" / "processed" / "bottle" / "train" / "good"
SPLIT_PATH = ROOT / "data" / "processed" / "bottle" / "calibration_split.csv"
OUTPUT_PATH = ROOT / "models" / "patchcore_memory_bank.pth"

IMAGE_SIZE = 256
BATCH_SIZE = 8
CANDIDATE_SIZE = 8192
BANK_SIZE = 1024
SEED = 42

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

    # Load only the training images from the calibration split
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

    # Initialize frozen feature extractor
    torch.manual_seed(SEED)

    feature_extractor = PatchCoreFeatureExtractor().to(DEVICE)
    feature_extractor.eval()

    extracted_features = []

    print("\nExtracting patch features...")

    with torch.no_grad():

        for images in tqdm(loader):

            images = images.to(DEVICE)

            feature_maps = feature_extractor(images)

            # [B, 384, 32, 32] -> [B*1024, 384]
            batch_features = feature_maps.permute(
                0, 2, 3, 1
            ).reshape(-1, 384)

            extracted_features.append(
                batch_features.cpu()
            )

    all_features = torch.cat(extracted_features, dim=0)

    print(f"\nAll patch features: {all_features.shape}")

    # Select a reproducible random candidate subset
    generator = torch.Generator().manual_seed(SEED)

    candidate_indices = torch.randperm(
        len(all_features),
        generator=generator
    )[:CANDIDATE_SIZE]

    candidate_features = all_features[candidate_indices]

    print(f"Candidate features: {candidate_features.shape}")

    # Greedy farthest-point coreset selection
    print("\nBuilding memory bank using greedy coreset...")

    memory_bank = greedy_coreset(
        candidate_features.to(DEVICE),
        bank_size=BANK_SIZE
    )

    memory_bank = memory_bank.cpu().contiguous()

    # Save checkpoint
    checkpoint = {
        "memory_bank": memory_bank,
        "feature_channels": 384,
        "image_size": IMAGE_SIZE,
        "candidate_size": CANDIDATE_SIZE,
        "bank_size": BANK_SIZE,
        "seed": SEED,
        "training_samples": len(image_paths)
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, OUTPUT_PATH)

    print("\nPATCHCORE TRAINING COMPLETED")
    print("----------------------------")
    print(f"Training images: {len(image_paths)}")
    print(f"Total patches: {len(all_features)}")
    print(f"Candidate patches: {len(candidate_features)}")
    print(f"Memory bank shape: {memory_bank.shape}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
