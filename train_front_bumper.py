from pathlib import Path
import shutil

from anomalib.data import Folder
from anomalib.engine import Engine
from anomalib.models import Patchcore


# ============================================================
# SETTINGS
# ============================================================

PART_NAME = "front_bumper"

DATA_ROOT = Path("patchcore_data_17") / PART_NAME
TRAIN_DIR = DATA_ROOT / "train" / "good"
TEST_DIR = DATA_ROOT / "test" / "good"

RESULTS_DIR = Path("patchcore_results_17") / PART_NAME

EXPECTED_TOTAL = 88
HOLDOUT_COUNT = 10


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("PATCHCORE TRAINING - FRONT BUMPER")
print("=" * 70)
print()

print(f"Part: {PART_NAME}")
print(f"Data folder: {DATA_ROOT.resolve()}")
print(f"Training folder: {TRAIN_DIR.resolve()}")
print(f"Test folder: {TEST_DIR.resolve()}")
print(f"Results folder: {RESULTS_DIR.resolve()}")
print()


# ============================================================
# CHECK TRAINING DATA
# ============================================================

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder does not exist:\n{TRAIN_DIR.resolve()}"
    )


image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG",
}


image_files = sorted(
    [
        p
        for p in TRAIN_DIR.iterdir()
        if p.is_file() and p.suffix in image_extensions
    ]
)


print(f"Training images currently found: {len(image_files)}")

if len(image_files) != EXPECTED_TOTAL:
    print(
        f"WARNING: Expected {EXPECTED_TOTAL} clean images, "
        f"but found {len(image_files)}."
    )
else:
    print("Dataset check: PASS")

print()


# ============================================================
# CREATE CLEAN HOLDOUT SET
# ============================================================

TEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Check whether holdout images already exist.
existing_test_images = sorted(
    [
        p
        for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix in image_extensions
    ]
)


if len(existing_test_images) == 0:

    print("Creating clean validation/test holdout...")
    print(f"Moving {HOLDOUT_COUNT} images from train/good to test/good.")
    print()

    if len(image_files) <= HOLDOUT_COUNT:
        raise RuntimeError(
            "Not enough training images to create a holdout set."
        )

    holdout_images = image_files[-HOLDOUT_COUNT:]

    for source_file in holdout_images:

        destination_file = TEST_DIR / source_file.name

        shutil.move(
            str(source_file),
            str(destination_file),
        )

        print(f"Moved: {source_file.name}")

    print()
    print("Holdout creation complete.")

else:

    print(
        f"Existing test/holdout images found: "
        f"{len(existing_test_images)}"
    )

print()


# ============================================================
# RECOUNT DATA
# ============================================================

train_images = sorted(
    [
        p
        for p in TRAIN_DIR.iterdir()
        if p.is_file() and p.suffix in image_extensions
    ]
)

test_images = sorted(
    [
        p
        for p in TEST_DIR.iterdir()
        if p.is_file() and p.suffix in image_extensions
    ]
)


print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

print(f"Training images : {len(train_images)}")
print(f"Test images : {len(test_images)}")
print(f"Total clean : {len(train_images) + len(test_images)}")

print()


if len(train_images) == 0:
    raise RuntimeError("No training images remain.")

if len(test_images) == 0:
    raise RuntimeError("No test images were created.")

print("Dataset split: PASS")
print()


# ============================================================
# CREATE ANOMALIB FOLDER DATAMODULE
# ============================================================

print("Creating Anomalib Folder datamodule...")

datamodule = Folder(
    name=PART_NAME,

    root=str(DATA_ROOT),

    # Training images
    normal_dir="train/good",

    # Clean validation/test images
    normal_test_dir="test/good",

    # There are no damaged training images.
    abnormal_dir=None,

    # Do not automatically split the training images.
    normal_split_ratio=0.0,

    extensions=(
        ".jpg",
        ".jpeg",
        ".png",
        ".JPG",
        ".JPEG",
        ".PNG",
    ),

    train_batch_size=8,
    eval_batch_size=8,

    # Windows CPU stability
    num_workers=0,

    # Use the explicit test/good folder.
    test_split_mode="from_dir",

    # Split the clean test images between validation/test.
    val_split_mode="same_as_test",
    val_split_ratio=0.5,
)


# ============================================================
# CREATE PATCHCORE MODEL
# ============================================================

print("Creating PatchCore model...")

model = Patchcore(
    backbone="wide_resnet50_2",

    layers=[
        "layer2",
        "layer3",
    ],

    coreset_sampling_ratio=0.1,

    num_neighbors=9,

    pre_trained=True,
)


# ============================================================
# CREATE ENGINE
# ============================================================

print("Creating Anomalib Engine...")

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

engine = Engine(
    default_root_dir=str(RESULTS_DIR)
)


# ============================================================
# TRAIN
# ============================================================

print()
print("=" * 70)
print("STARTING PATCHCORE TRAINING")
print("=" * 70)
print()

engine.fit(
    model=model,
    datamodule=datamodule,
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("PATCHCORE TRAINING FINISHED")
print("=" * 70)
print()

print(f"Part: {PART_NAME}")
print(f"Training images: {len(train_images)}")
print(f"Validation/test images: {len(test_images)}")
print()

print("Results saved under:")
print(RESULTS_DIR.resolve())
print()

print("Front bumper PatchCore training is complete.")