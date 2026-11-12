
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from src.inference.inference_engine import ForgeVisionInference


normal_dir = ROOT / "data" / "processed" / "bottle" / "test" / "good"
defective_dir = (
    ROOT / "data" / "processed" / "bottle" / "test"
    / "defective" / "images"
)

normal_images = sorted(normal_dir.glob("*.png"))
defective_images = sorted(defective_dir.glob("*.png"))

if not normal_images or not defective_images:
    raise FileNotFoundError("Test images were not found.")

engine = ForgeVisionInference()

test_samples = [
    ("NORMAL", normal_images[0]),
    ("DEFECTIVE", defective_images[0])
]

for label, image_path in test_samples:

    print(f"\nTesting {label}: {image_path.name}")

    output = engine.predict(image_path)

    for model_name, result in output["results"].items():

        print(
            f"{model_name}: "
            f"Score={result['score']:.6f}, "
            f"Map shape={result['anomaly_map'].shape}, "
            f"Time={result['inference_ms']:.2f} ms"
        )

print("\nINFERENCE TEST COMPLETED")
