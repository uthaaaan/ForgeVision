
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PatchCoreFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)

        self.stem = nn.Sequential(
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
        x = self.stem(x)
        x = self.layer1(x)

        layer2_features = self.layer2(x)
        layer3_features = self.layer3(layer2_features)

        layer3_features = F.interpolate(
            layer3_features,
            size=layer2_features.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        return torch.cat(
            [layer2_features, layer3_features],
            dim=1
        )


def greedy_coreset(features, bank_size=512):
    features = F.normalize(features.float(), dim=1)

    n_samples = features.shape[0]

    if bank_size >= n_samples:
        return features

    selected_indices = []
    min_distances = torch.full(
        (n_samples,),
        float("inf"),
        device=features.device
    )

    current_index = 0

    for _ in range(bank_size):
        selected_indices.append(current_index)

        current_feature = features[current_index].unsqueeze(0)

        distances = torch.cdist(
            features,
            current_feature
        ).squeeze(1)

        min_distances = torch.minimum(
            min_distances,
            distances
        )

        current_index = torch.argmax(min_distances).item()

    return features[selected_indices]
