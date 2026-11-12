from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

RESULTS_PATH = ROOT / "reports" / "baseline_results.csv"

padim_result = {
    "Model": "PaDiM",
    "Category": "bottle",
    "Image AUROC": 0.9952,
    "Pixel AUROC": 0.9824,
    "Average Inference (ms)": 31.10,
    "Normal Mean Error": 149.988800,
    "Defective Mean Error": 880.963562,
    "Best Validation Loss": None,
    "Best Epoch": None
}

results = pd.read_csv(RESULTS_PATH)

# Replace an existing PaDiM row if the script is rerun.
results = results[results["Model"] != "PaDiM"]

results = pd.concat(
    [results, pd.DataFrame([padim_result])],
    ignore_index=True
)

results.to_csv(RESULTS_PATH, index=False)

print("\nMODEL COMPARISON")
print("----------------")
print(
    results[
        ["Model", "Image AUROC", "Pixel AUROC", "Average Inference (ms)"]
    ].to_string(index=False)
)

print(f"\nUpdated: {RESULTS_PATH}")