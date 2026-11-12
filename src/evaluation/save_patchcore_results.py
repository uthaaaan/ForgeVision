from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "reports" / "baseline_results.csv"

patchcore_result = {
    "Model": "PatchCore",
    "Category": "bottle",
    "Image AUROC": 1.0000,
    "Pixel AUROC": 0.9815,
    "Average Inference (ms)": 10.14,
    "Normal Mean Error": 0.593723,
    "Defective Mean Error": 0.933960,
    "Best Validation Loss": None,
    "Best Epoch": None
}

results = pd.read_csv(RESULTS_PATH)

# Replace an existing PatchCore row if the script is rerun.
results = results[results["Model"] != "PatchCore"]

results = pd.concat(
    [results, pd.DataFrame([patchcore_result])],
    ignore_index=True
)

results.to_csv(RESULTS_PATH, index=False)

print("\nFINAL MODEL COMPARISON")
print("----------------------")
print(
    results[
        ["Model", "Image AUROC", "Pixel AUROC",
         "Average Inference (ms)"]
    ].to_string(index=False)
)

print(f"\nUpdated: {RESULTS_PATH}")