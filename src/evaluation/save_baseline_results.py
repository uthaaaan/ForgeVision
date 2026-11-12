from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "reports" / "baseline_results.csv"

results = [{
    "Model": "Convolutional Autoencoder",
    "Category": "bottle",
    "Image AUROC": 0.8929,
    "Pixel AUROC": 0.7272,
    "Average Inference (ms)": 30.06,
    "Normal Mean Error": 0.00046345,
    "Defective Mean Error": 0.00058395,
    "Best Validation Loss": 0.00040094,
    "Best Epoch": 48
}]

pd.DataFrame(results).to_csv(OUTPUT, index=False)

print("BASELINE RESULTS SAVED")
print(pd.read_csv(OUTPUT).to_string(index=False))
print(f"\nSaved to: {OUTPUT}")