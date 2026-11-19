from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.data import PredictDataset


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = (
    Path("patchcore_models_17")
    / "hood"
    / "model.ckpt"
)

IMAGE_PATH = (
    Path("damaged_test_crops")
    / "hood.jpg"
)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 80)
print("PATCHCORE DAMAGED HOOD TEST")
print("=" * 80)

if not MODEL_PATH.exists():
    print("❌ Hood model not found:")
    print(MODEL_PATH.resolve())
    raise SystemExit(1)

if not IMAGE_PATH.exists():
    print("❌ Damaged hood image not found:")
    print(IMAGE_PATH.resolve())
    raise SystemExit(1)

print()
print("Model:")
print(MODEL_PATH.resolve())

print()
print("Test image:")
print(IMAGE_PATH.resolve())


# ============================================================
# CREATE PATCHCORE MODEL
# ============================================================

print()
print("Creating PatchCore model...")

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
    pre_trained=True,
)

engine = Engine()


# ============================================================
# CREATE PREDICTION DATASET
# ============================================================

print()
print("Preparing test image...")

dataset = PredictDataset(
    path=str(IMAGE_PATH),
    image_size=(256, 256),
)


# ============================================================
# RUN INFERENCE
# ============================================================

print()
print("Running damaged hood test...")
print("CPU inference may take some time.")

predictions = engine.predict(
    model=model,
    dataset=dataset,
    ckpt_path=str(MODEL_PATH),
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print()
print("=" * 80)
print("DAMAGED HOOD RESULT")
print("=" * 80)

if predictions is None:
    print("❌ No prediction returned.")
    raise SystemExit(1)

for prediction in predictions:

    image_path = getattr(
        prediction,
        "image_path",
        str(IMAGE_PATH)
    )

    pred_score = getattr(
        prediction,
        "pred_score",
        None
    )

    pred_label = getattr(
        prediction,
        "pred_label",
        None
    )

    print()
    print(f"Image : {image_path}")
    print(f"Anomaly : {pred_score}")
    print(f"Label : {pred_label}")

    if pred_score is not None:
        try:
            score = float(pred_score)
            print(f"Score : {score:.6f}")
        except Exception:
            pass

print()
print("=" * 80)
print("TEST FINISHED")
print("=" * 80)