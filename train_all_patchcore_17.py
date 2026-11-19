from pathlib import Path
import traceback

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# FIXED 17-PART SYSTEM
# ============================================================

PARTS = [
    "front_bumper",
    "rear_bumper",
    "front_door",
    "rear_door",
    "front_window",
    "rear_window",
    "wheel",
    "front_fender",
    "quarter_panel",
    "hood",
    "trunk_lid",
    "headlight",
    "taillight",
    "side_mirror",
    "windshield_front",
    "windshield_back",
    "license_plate",
]


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = PROJECT_ROOT / "patchcore_data_17_split"

RESULTS_ROOT = PROJECT_ROOT / "patchcore_results_17"

# Never overwrite the existing production checkpoints.
OLD_MODELS_ROOT = PROJECT_ROOT / "patchcore_models_17"


# ============================================================
# TRAINING SETTINGS
# ============================================================

IMAGE_SIZE = (256, 256)

TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
NUM_WORKERS = 0

BACKBONE = "wide_resnet50_2"
LAYERS = ["layer2", "layer3"]
CORESET_SAMPLING_RATIO = 0.1
NUM_NEIGHBORS = 9


# ============================================================
# HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".JPG", ".JPEG", ".PNG",
}


def get_images(folder):
    if not folder.is_dir():
        return []

    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix in IMAGE_EXTENSIONS
    )


def verify_dataset():
    print("\nVERIFYING UPDATED DATASET")
    print("=" * 70)

    errors = []

    for part in PARTS:
        train_dir = DATA_ROOT / part / "train" / "good"
        test_dir = DATA_ROOT / part / "test" / "good"

        if not train_dir.is_dir():
            errors.append(f"Missing training folder: {train_dir}")
            continue

        if not test_dir.is_dir():
            errors.append(f"Missing test folder: {test_dir}")
            continue

        train_images = get_images(train_dir)
        test_images = get_images(test_dir)

        if not train_images:
            errors.append(f"No training images for {part}")

        if not test_images:
            errors.append(f"No test images for {part}")

        print(
            f"{part:20s} "
            f"train={len(train_images):4d} "
            f"test={len(test_images):3d}"
        )

    print("=" * 70)

    if errors:
        for error in errors:
            print("ERROR:", error)

        raise RuntimeError(
            "Dataset verification failed. No models were trained."
        )

    print("All 17 categories passed the folder checks.")


# ============================================================
# TRAIN ONE PART
# ============================================================

def train_part(part_name):
    print("\n" + "=" * 70)
    print(f"TRAINING PATCHCORE: {part_name}")
    print("=" * 70)

    part_root = DATA_ROOT / part_name
    train_dir = part_root / "train" / "good"
    test_dir = part_root / "test" / "good"

    results_dir = RESULTS_ROOT / part_name

    # Verify the existing split without moving any images.
    train_images = get_images(train_dir)
    test_images = get_images(test_dir)

    if not train_images or not test_images:
        raise RuntimeError(
            f"Empty training or test folder for {part_name}"
        )

    print(f"Training images: {len(train_images)}")
    print(f"Test images: {len(test_images)}")
    print(f"Results folder: {results_dir}")

    # --------------------------------------------------------
    # Anomalib data module
    # --------------------------------------------------------

    datamodule = Folder(
        name=part_name,
        root=str(part_root),
        normal_dir="train/good",
        normal_test_dir="test/good",
        abnormal_dir=None,
        normal_split_ratio=0.0,
        extensions=(
            ".jpg",
            ".jpeg",
            ".png",
            ".JPG",
            ".JPEG",
            ".PNG",
        ),
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_batch_size=EVAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
        val_split_ratio=0.5,
    )

    # --------------------------------------------------------
    # PatchCore model
    # --------------------------------------------------------

    model = Patchcore(
        backbone=BACKBONE,
        layers=LAYERS,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        num_neighbors=NUM_NEIGHBORS,
        pre_trained=True,
    )

    # --------------------------------------------------------
    # CPU engine
    # --------------------------------------------------------

    results_dir.mkdir(parents=True, exist_ok=True)

    engine = Engine(
        default_root_dir=str(results_dir),
        accelerator="cpu",
        devices=1,
    )

    # --------------------------------------------------------
    # Train and evaluate
    # --------------------------------------------------------

    engine.fit(
        model=model,
        datamodule=datamodule,
    )

    print(f"\nFINISHED: {part_name}")
    print(f"Results saved under: {results_dir}")

    return True


# ============================================================
# MAIN
# ============================================================


# ============================================================
# RESUME TRAINING FROM SIDE_MIRROR
# ============================================================

def main():

    print("=" * 70)
    print("PATCHCORE TRAINING - RESUMING FROM SIDE_MIRROR")
    print("=" * 70)

    print(f"Dataset: {DATA_ROOT}")
    print(f"Results: {RESULTS_ROOT}")

    # Verify dataset
    if not DATA_ROOT.is_dir():
        raise FileNotFoundError(
            f"Updated dataset not found: {DATA_ROOT}"
        )

    verify_dataset()

    # Do not overwrite or delete existing results.
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    # Only the four remaining parts
    REMAINING_PARTS = [
        "side_mirror",
        "windshield_front",
        "windshield_back",
        "license_plate",
    ]

    successful = []
    failed = []

    print("\nRemaining parts:")
    for i, part in enumerate(REMAINING_PARTS, start=1):
        print(f"{i}. {part}")

    # Train remaining parts sequentially
    for index, part in enumerate(REMAINING_PARTS, start=1):

        print("\n" + "=" * 70)
        print(
            f"RESUMING PART {index}/{len(REMAINING_PARTS)}: {part}"
        )
        print("=" * 70)

        try:

            train_part(part)

            successful.append(part)

            print(f"\nSUCCESSFULLY COMPLETED: {part}")

        except Exception as error:

            failed.append((part, str(error)))

            print(f"\nTRAINING FAILED: {part}")
            print(error)

            traceback.print_exc()

            print("\nContinuing to the next part.")

    # Final report
    print("\n" + "=" * 70)
    print("PATCHCORE RESUME TRAINING SUMMARY")
    print("=" * 70)

    print(
        f"Successfully completed: "
        f"{len(successful)}/{len(REMAINING_PARTS)}"
    )

    print(f"Failed: {len(failed)}")

    if successful:
        print("\nSUCCESSFUL PARTS:")

        for part in successful:
            print("[OK]", part)

    if failed:
        print("\nFAILED PARTS:")

        for part, error in failed:
            print("[FAILED]", part)
            print("Reason:", error)

    print("\nExisting and new results are stored at:")
    print(RESULTS_ROOT)

    if not failed:
        print("\nALL FOUR REMAINING PARTS COMPLETED SUCCESSFULLY.")

    print("=" * 70)


if __name__ == "__main__":
    main()
