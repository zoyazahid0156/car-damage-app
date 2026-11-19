from pathlib import Path
from ultralytics import YOLO

PROJECT = Path(
    r"C:\Users\zoya.zahid\Documents\car dataset project"
)

MODEL_PATH = (
    PROJECT
    / "runs"
    / "detect"
    / "detector_results_17"
    / "car_parts_283"
    / "weights"
    / "best.pt"
)

DATA_YAML = PROJECT / "yolo_dataset_17" / "dataset.yaml"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

print("Loading trained YOLO model...")
model = YOLO(str(MODEL_PATH))

print("\nRunning validation on the 50 validation images...")
metrics = model.val(
    data=str(DATA_YAML),
    split="val",
    imgsz=640,
    batch=1,
    workers=0,
    device="cpu",
    plots=False,
)

print("\n========== VALIDATION SUMMARY ==========")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print("========================================")