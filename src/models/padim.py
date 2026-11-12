
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.layer0 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        for parameter in self.parameters():
            parameter.requires_grad = False

        self.eval()

    def forward(self, x):
        x = self.layer0(x)

        features1 = self.layer1(x)
        features2 = self.layer2(features1)
        features3 = self.layer3(features2)

        target_size = features1.shape[-2:]

        features2 = F.interpolate(
            features2,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        features3 = F.interpolate(
            features3,
            size=target_size,
            mode="bilinear",
            align_corners=False
        )

        features = torch.cat(
            [features1, features2, features3],
            dim=1
        )

        return features


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ResNetFeatureExtractor().to(device)
    model.eval()

    dummy_input = torch.randn(
        2, 3, 256, 256
    ).to(device)

    with torch.no_grad():
        output = model(dummy_input)

    print("\nPADIM FEATURE EXTRACTOR VALIDATION")
    print("---------------------------------")
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape}")
    print(f"Feature shape: {output.shape}")
    print(f"Feature channels: {output.shape[1]}")
    print(f"Feature map size: {output.shape[-2:]}")

    assert output.shape == (2, 448, 64, 64)

    print("\nFeature extractor validation successful!")
