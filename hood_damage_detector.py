from pathlib import Path

from anomalib.engine import Engine
from anomalib.models import Patchcore


# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_DIR = Path(__file__).resolve().parent

CHECKPOINT = (
    PROJECT_DIR
    / "patchcore_results"
    / "hood"
    / "Patchcore"
    / "part_hood"
    / "v0"
    / "weights"
    / "lightning"
    / "model.ckpt"
)

# Provisional threshold calibrated from current clean data.
THRESHOLD = 0.55


# =========================================================
# CREATE MODEL
# =========================================================

model = Patchcore(
    backbone="wide_resnet50_2",
    layers=["layer2", "layer3"],
    pre_trained=True,
    coreset_sampling_ratio=0.1,
    num_neighbors=9,
    precision="float32",
)


# =========================================================
# CREATE ENGINE
# =========================================================

engine = Engine(
    accelerator="cpu",
    devices=1,
    logger=False,
)


# =========================================================
# FUNCTION
# =========================================================

def detect_hood_damage(image_path):
    """
    Detect whether a hood image appears damaged.

    Returns:
        dictionary containing:
        - part
        - status
        - anomaly_score
        - threshold
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"PatchCore checkpoint not found: {CHECKPOINT}"
        )

    # -----------------------------------------------------
    # Run PatchCore
    # -----------------------------------------------------

    predictions = engine.predict(
        model=model,
        data_path=str(image_path),
        ckpt_path=str(CHECKPOINT),
        return_predictions=True,
    )

    prediction = predictions[0]

    # -----------------------------------------------------
    # Extract anomaly score
    # -----------------------------------------------------

    anomaly_score = float(
        prediction.pred_score[0]
    )

    # -----------------------------------------------------
    # Apply our calibrated threshold
    # -----------------------------------------------------

    if anomaly_score >= THRESHOLD:
        status = "DAMAGED"
    else:
        status = "OK"

    # -----------------------------------------------------
    # Return clean result
    # -----------------------------------------------------

    return {
        "part": "hood",
        "status": status,
        "anomaly_score": anomaly_score,
        "threshold": THRESHOLD,
    }


# =========================================================
# TEST WHEN RUN DIRECTLY
# =========================================================

if __name__ == "__main__":

    test_image = (
        PROJECT_DIR
        / "damaged_test_crops"
        / "hood.jpg"
    )

    result = detect_hood_damage(test_image)

    print()
    print("=" * 60)
    print("HOOD DAMAGE DETECTOR")
    print("=" * 60)
    print()

    print(f"Part: {result['part']}")
    print(f"Status: {result['status']}")
    print(f"Anomaly score: {result['anomaly_score']:.4f}")
    print(f"Threshold: {result['threshold']:.4f}")

    print()
    print("=" * 60)