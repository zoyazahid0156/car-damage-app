from pathlib import Path
from ultralytics import YOLO


# ============================================================
# 17-CLASS CAR PART DETECTOR TRAINING
# ============================================================

PROJECT_ROOT = Path.cwd()

DATASET_YAML = PROJECT_ROOT / "detector_dataset_17" / "data.yaml"

# YOLO model
BASE_MODEL = "yolo26s.pt"

# Where training results will be stored
PROJECT_DIR = PROJECT_ROOT / "detector_results_17"

# Training name
RUN_NAME = "car_parts_17"


print("=" * 80)
print("TRAINING 17-CLASS CAR PART DETECTOR")
print("=" * 80)

print()
print("Dataset:")
print(DATASET_YAML)

print()
print("Base model:")
print(BASE_MODEL)

print()
print("Output:")
print(PROJECT_DIR)

print("=" * 80)


# ------------------------------------------------------------
# Check dataset
# ------------------------------------------------------------

if not DATASET_YAML.exists():
    print()
    print("ERROR: data.yaml was not found.")
    print()
    print("Expected:")
    print(DATASET_YAML)
    print()
    raise FileNotFoundError(DATASET_YAML)


# ------------------------------------------------------------
# Load YOLO model
# ------------------------------------------------------------

print()
print("Loading YOLO model...")
print()

model = YOLO(BASE_MODEL)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------

print()
print("=" * 80)
print("STARTING TRAINING")
print("=" * 80)
print()

results = model.train(
    data=str(DATASET_YAML),

    # Number of epochs
    epochs=100,

    # CPU-friendly image size
    imgsz=640,

    # Small batch for CPU/RAM
    batch=4,

    # CPU
    device="cpu",

    # Number of dataloader workers
    workers=0,

    # Training output
    project=str(PROJECT_DIR),
    name=RUN_NAME,

    # Save checkpoints
    save=True,
    save_period=10,

    # Use pretrained weights
    pretrained=True,

    # Reproducibility
    seed=42,

    # Cache disabled to reduce RAM/disk usage
    cache=False,

    # Automatically choose reasonable augmentation
    amp=False,

    # Close mosaic near the end of training
    close_mosaic=10,

    # Validation during training
    val=True,

    # Verbose training output
    verbose=True,
)


# ------------------------------------------------------------
# Training finished
# ------------------------------------------------------------

print()
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)

RUN_DIR = PROJECT_DIR / RUN_NAME
BEST_MODEL = RUN_DIR / "weights" / "best.pt"
LAST_MODEL = RUN_DIR / "weights" / "last.pt"

print()
print("Training results:")
print(RUN_DIR)

print()
print("Best model:")
print(BEST_MODEL)

print()
print("Last model:")
print(LAST_MODEL)

print()

if BEST_MODEL.exists():
    print("✅ BEST MODEL CREATED")
else:
    print("⚠️ best.pt was not found.")

print()
print("=" * 80)