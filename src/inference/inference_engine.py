
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.autoencoder import ConvAutoencoder
from src.models.padim import ResNetFeatureExtractor
from src.models.patchcore import PatchCoreFeatureExtractor


ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 256

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class InferenceEngine:

    def __init__(self):

        self.device = DEVICE

        self.basic_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor()
        ])

        self.imagenet_transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.models = {}
        self.thresholds = {}

        self.load_models()

    def load_models(self):

        print(f"Loading ForgeVision models on {self.device}...")

        # -------------------------
        # 1. Autoencoder
        # -------------------------

        ae_checkpoint = torch.load(
            ROOT / "models" / "autoencoder_best.pth",
            map_location=self.device,
            weights_only=False
        )

        autoencoder = ConvAutoencoder().to(self.device)

        autoencoder.load_state_dict(
            ae_checkpoint["model_state_dict"]
        )

        autoencoder.eval()

        self.models["autoencoder"] = autoencoder

        # -------------------------
        # 2. PaDiM
        # -------------------------

        padim_checkpoint = torch.load(
            ROOT / "models" / "padim_parameters.pth",
            map_location="cpu",
            weights_only=False
        )

        padim_extractor = ResNetFeatureExtractor().to(self.device)
        padim_extractor.eval()

        self.models["padim_extractor"] = padim_extractor

        self.models["selected_channels"] = (
            padim_checkpoint["selected_channels"].to(self.device)
        )

        self.models["padim_mean"] = (
            padim_checkpoint["mean"].to(self.device)
        )

        self.models["padim_inverse_covariance"] = (
            padim_checkpoint["inverse_covariance"].to(self.device)
        )

        # -------------------------
        # 3. PatchCore
        # -------------------------

        patchcore_checkpoint = torch.load(
            ROOT / "models" / "patchcore_memory_bank.pth",
            map_location="cpu",
            weights_only=False
        )

        patchcore_extractor = PatchCoreFeatureExtractor().to(self.device)
        patchcore_extractor.eval()

        self.models["patchcore_extractor"] = patchcore_extractor

        memory_bank = patchcore_checkpoint["memory_bank"].to(
            self.device
        )

        self.models["memory_bank"] = F.normalize(
            memory_bank,
            dim=1
        )

        # -------------------------
        # 4. Calibration thresholds
        # -------------------------

        threshold_path = (
            ROOT / "models" / "calibration_thresholds.json"
        )

        with open(threshold_path, "r") as file:
            threshold_data = json.load(file)

        self.thresholds = threshold_data["thresholds"]

        print("All models and thresholds loaded successfully.")

        for name, threshold in self.thresholds.items():
            print(f"{name}: threshold = {threshold:.8f}")

    def _classify(self, model_name, score):

        threshold = self.thresholds[model_name]

        prediction = (
            "Defective" if score > threshold else "Normal"
        )

        return {
            "score": float(score),
            "threshold": float(threshold),
            "prediction": prediction
        }

    def predict(self, image):

        if not isinstance(image, Image.Image):
            image = Image.open(image)

        image = image.convert("RGB")

        basic_image = (
            self.basic_transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        normalized_image = (
            self.imagenet_transform(image)
            .unsqueeze(0)
            .to(self.device)
        )

        results = {}

        # -------------------------
        # Autoencoder inference
        # -------------------------

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            reconstruction = self.models["autoencoder"](
                basic_image
            )

            error_map = torch.mean(
                (basic_image - reconstruction) ** 2,
                dim=1
            )

            ae_score = error_map.max().item()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        ae_time = (time.perf_counter() - start_time) * 1000

        ae_result = self._classify(
            "autoencoder",
            ae_score
        )

        ae_result.update({
            "anomaly_map": error_map[0].cpu().numpy(),
            "inference_ms": ae_time
        })

        results["autoencoder"] = ae_result

        # -------------------------
        # PaDiM inference
        # -------------------------

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            features = self.models["padim_extractor"](
                normalized_image
            )

            features = features[
                :,
                self.models["selected_channels"]
            ]

            batch, channels, height, width = features.shape

            features = features.reshape(
                batch,
                channels,
                height * width
            ).permute(0, 2, 1)

            difference = (
                features
                - self.models["padim_mean"].T.unsqueeze(0)
            )

            squared_distance = torch.einsum(
                "bnc,ncd,bnd->bn",
                difference,
                self.models["padim_inverse_covariance"],
                difference
            )

            anomaly_map = torch.sqrt(
                squared_distance.clamp(min=0)
            ).reshape(
                batch, 1, height, width
            )

            anomaly_map = F.interpolate(
                anomaly_map,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear",
                align_corners=False
            )

            padim_score = anomaly_map.max().item()

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        padim_time = (time.perf_counter() - start_time) * 1000

        padim_result = self._classify(
            "padim",
            padim_score
        )

        padim_result.update({
            "anomaly_map": anomaly_map[0, 0].cpu().numpy(),
            "inference_ms": padim_time
        })

        results["padim"] = padim_result

        # -------------------------
        # PatchCore inference
        # -------------------------

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.no_grad():

            feature_maps = self.models["patchcore_extractor"](
                normalized_image
            )

            batch, channels, height, width = feature_maps.shape

            patch_features = feature_maps.permute(
                0, 2, 3, 1
            ).reshape(-1, channels)

            patch_features = F.normalize(
                patch_features,
                dim=1
            )

            distances = torch.cdist(
                patch_features,
                self.models["memory_bank"]
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

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        patchcore_time = (
            time.perf_counter() - start_time
        ) * 1000

        patchcore_result = self._classify(
            "patchcore",
            patchcore_score
        )

        patchcore_result.update({
            "anomaly_map": patch_map[0, 0].cpu().numpy(),
            "inference_ms": patchcore_time
        })

        results["patchcore"] = patchcore_result

        return {
            "original_image": image,
            "results": results
        }
# Backward compatibility with existing scripts
ForgeVisionInference = InferenceEngine