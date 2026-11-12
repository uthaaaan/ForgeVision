
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.models.autoencoder import ConvAutoencoder
from src.models.padim import ResNetFeatureExtractor
from src.models.patchcore import PatchCoreFeatureExtractor


# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256
NUM_SAMPLES = 5

DATA_DIR = ROOT / "data" / "processed" / "bottle"
MODELS_DIR = ROOT / "models"
OUTPUT_PATH = ROOT / "reports" / "multi_sample_anomaly_comparison.png"

resize_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

feature_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def normalize_map(anomaly_map):
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    minimum = anomaly_map.min()
    maximum = anomaly_map.max()

    return (anomaly_map - minimum) / (
        maximum - minimum + 1e-8
    )


# Load models
print("Loading models...")

autoencoder_checkpoint = torch.load(
    MODELS_DIR / "autoencoder_best.pth",
    map_location=DEVICE,
    weights_only=True
)

autoencoder = ConvAutoencoder().to(DEVICE)
autoencoder.load_state_dict(
    autoencoder_checkpoint["model_state_dict"]
)
autoencoder.eval()

padim = ResNetFeatureExtractor().to(DEVICE)
padim.eval()

padim_params = torch.load(
    MODELS_DIR / "padim_parameters.pth",
    map_location="cpu",
    weights_only=True
)

padim_mean = padim_params["mean"].to(DEVICE)
padim_inverse_cov = padim_params["inverse_covariance"].to(DEVICE)
padim_channels = padim_params["selected_channels"].to(DEVICE)

padim_height = padim_params["feature_height"]
padim_width = padim_params["feature_width"]

patchcore_checkpoint = torch.load(
    MODELS_DIR / "patchcore_memory_bank.pth",
    map_location=DEVICE,
    weights_only=True
)

memory_bank = patchcore_checkpoint["memory_bank"].to(DEVICE)
memory_bank = F.normalize(memory_bank.float(), dim=1)

patchcore = PatchCoreFeatureExtractor().to(DEVICE)
patchcore.eval()


def generate_maps(image_path):
    image = Image.open(image_path).convert("RGB")

    image_tensor = resize_transform(image).unsqueeze(0).to(DEVICE)
    feature_tensor = feature_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        # Autoencoder
        reconstructed = autoencoder(image_tensor)

        autoencoder_map = torch.mean(
            (image_tensor - reconstructed) ** 2,
            dim=1
        ).squeeze().cpu().numpy()

        # PaDiM
        features = padim(feature_tensor)
        features = features[:, padim_channels]

        features = features.squeeze(0).reshape(
            len(padim_channels), -1
        )

        difference = features.T - padim_mean.T

        distance = torch.einsum(
            "nd,ndk,nk->n",
            difference,
            padim_inverse_cov,
            difference
        )

        padim_map = distance.reshape(
            1, 1, padim_height, padim_width
        )

        padim_map = F.interpolate(
            padim_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

        # PatchCore
        features = patchcore(feature_tensor)

        patches = features.permute(
            0, 2, 3, 1
        ).reshape(-1, 384)

        patches = F.normalize(patches.float(), dim=1)

        distance_chunks = []

        for start in range(0, len(patches), 256):
            query = patches[start:start + 256]

            distances = torch.cdist(query, memory_bank)

            distance_chunks.append(
                distances.min(dim=1).values
            )

        distances = torch.cat(distance_chunks)

        patchcore_map = distances.reshape(
            1, 1, features.shape[2], features.shape[3]
        )

        patchcore_map = F.interpolate(
            patchcore_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        ).squeeze().cpu().numpy()

    return (
        image.resize((IMAGE_SIZE, IMAGE_SIZE)),
        normalize_map(autoencoder_map),
        normalize_map(padim_map),
        normalize_map(patchcore_map)
    )


# Select defective samples
defect_dir = DATA_DIR / "test" / "defective" / "images"
mask_dir = DATA_DIR / "test" / "defective" / "masks"

defect_images = sorted(defect_dir.glob("*.png"))

if len(defect_images) < NUM_SAMPLES:
    raise RuntimeError("Not enough defective images.")

selected_images = defect_images[:NUM_SAMPLES]

print(f"Device: {DEVICE}")
print(f"Selected samples: {len(selected_images)}")


# Create comparison figure
fig, axes = plt.subplots(
    NUM_SAMPLES,
    5,
    figsize=(17, 4 * NUM_SAMPLES)
)

column_titles = [
    "Original Image",
    "Ground Truth",
    "Autoencoder",
    "PaDiM",
    "PatchCore"
]

for column, title in enumerate(column_titles):
    axes[0, column].set_title(
        title,
        fontsize=12,
        fontweight="bold"
    )

for row, image_path in enumerate(selected_images):

    print(f"Processing {row + 1}/{NUM_SAMPLES}: {image_path.name}")

    mask_path = mask_dir / f"{image_path.stem}.png"

    if not mask_path.exists():
        raise FileNotFoundError(mask_path)

    original, ae_map, padim_map, patchcore_map = generate_maps(
        image_path
    )

    ground_truth = Image.open(mask_path).convert("L")
    ground_truth = ground_truth.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.NEAREST
    )

    ground_truth = np.array(ground_truth) > 0

    axes[row, 0].imshow(original)

    axes[row, 1].imshow(original)
    axes[row, 1].imshow(
        ground_truth,
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1
    )

    axes[row, 2].imshow(original)
    axes[row, 2].imshow(
        ae_map,
        cmap="inferno",
        alpha=0.60,
        vmin=0,
        vmax=1
    )

    axes[row, 3].imshow(original)
    axes[row, 3].imshow(
        padim_map,
        cmap="inferno",
        alpha=0.60,
        vmin=0,
        vmax=1
    )

    axes[row, 4].imshow(original)
    axes[row, 4].imshow(
        patchcore_map,
        cmap="inferno",
        alpha=0.60,
        vmin=0,
        vmax=1
    )

    for ax in axes[row]:
        ax.axis("off")

    axes[row, 0].set_ylabel(
        f"Sample {row + 1}",
        fontsize=10
    )

fig.suptitle(
    "ForgeVision: Multi-Sample Anomaly Localization Comparison",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.98])

fig.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\nMULTI-SAMPLE COMPARISON COMPLETE")
print(f"Saved to: {OUTPUT_PATH}")
