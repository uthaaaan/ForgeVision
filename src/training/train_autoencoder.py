
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.models.autoencoder import ConvAutoencoder


DATA_DIR = ROOT / "data" / "processed" / "bottle" / "train" / "good"
SPLIT_FILE = ROOT / "data" / "processed" / "bottle" / "calibration_split.csv"
CHECKPOINT_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

EPOCHS = 50
BATCH_SIZE = 8
LEARNING_RATE = 0.001
IMAGE_SIZE = 256
SEED = 42


class NormalImageDataset(Dataset):
    def __init__(self, filenames, transform):
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        image_path = DATA_DIR / self.filenames[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)


def main():
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split_df = pd.read_csv(SPLIT_FILE)
    train_filenames = split_df.loc[
        split_df["split"] == "train", "filename"
    ].tolist()

    print(f"Training images: {len(train_filenames)}")
    print("Calibration images excluded from training: 42")

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = NormalImageDataset(train_filenames, transform)

    generator = torch.Generator().manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = ConvAutoencoder().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = []

    print("\nStarting Autoencoder training...")
    print(f"Epochs: {EPOCHS}")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for images in progress:
            images = images.to(device)

            optimizer.zero_grad()

            reconstructed = model(images)
            loss = criterion(reconstructed, images)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            progress.set_postfix(loss=f"{loss.item():.6f}")

        epoch_loss = running_loss / len(dataset)
        history.append({
            "epoch": epoch + 1,
            "training_loss": epoch_loss,
        })

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Training Loss: {epoch_loss:.8f}"
        )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint_path = CHECKPOINT_DIR / "autoencoder_best.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "epochs": EPOCHS,
        "training_samples": len(dataset),
        "image_size": IMAGE_SIZE,
        "seed": SEED,
    }, checkpoint_path)

    history_path = REPORT_DIR / "autoencoder_training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)

    print("\nAUTOENCODER TRAINING COMPLETE")
    print("--------------------------------")
    print(f"Training images: {len(dataset)}")
    print(f"Epochs completed: {EPOCHS}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")


if __name__ == "__main__":
    main()
