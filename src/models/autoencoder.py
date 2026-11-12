
import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # Encoder: compress the input image
        self.encoder = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder: reconstruct the input image
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                256, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 3, kernel_size=4, stride=2, padding=1
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)

        return reconstructed


if __name__ == "__main__":

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = ConvAutoencoder().to(device)

    dummy_input = torch.randn(8, 3, 256, 256).to(device)

    model.eval()

    with torch.no_grad():
        encoded = model.encoder(dummy_input)
        reconstructed = model(dummy_input)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("AUTOENCODER ARCHITECTURE VALIDATION")
    print("-----------------------------------")
    print("Device:", device)
    print("Input shape:", dummy_input.shape)
    print("Encoded shape:", encoded.shape)
    print("Reconstructed shape:", reconstructed.shape)
    print("Total parameters:", f"{total_parameters:,}")
    print("Output minimum:", reconstructed.min().item())
    print("Output maximum:", reconstructed.max().item())

    assert reconstructed.shape == dummy_input.shape
    assert reconstructed.min() >= 0
    assert reconstructed.max() <= 1

    print("\nArchitecture validation successful!")
