
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

HISTORY_PATH = ROOT / "reports" / "autoencoder_training_history.csv"
OUTPUT_PATH = ROOT / "reports" / "autoencoder_loss_curve.png"

history = pd.read_csv(HISTORY_PATH)

best_index = history["val_loss"].idxmin()
best_epoch = int(history.loc[best_index, "epoch"])
best_loss = history.loc[best_index, "val_loss"]

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    history["epoch"],
    history["train_loss"],
    label="Training Loss",
    linewidth=2
)

ax.plot(
    history["epoch"],
    history["val_loss"],
    label="Validation Loss",
    linewidth=2
)

ax.scatter(
    best_epoch,
    best_loss,
    marker="o",
    s=60,
    label=f"Best Validation Loss: {best_loss:.6f}"
)

ax.set_title("Autoencoder Training Performance")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=200)
plt.show()

print(f"Best epoch: {best_epoch}")
print(f"Best validation loss: {best_loss:.6f}")
print(f"Graph saved: {OUTPUT_PATH}")
