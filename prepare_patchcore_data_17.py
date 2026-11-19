from pathlib import Path
import shutil

PROJECT_ROOT = Path.cwd()

SOURCE_ROOT = PROJECT_ROOT / "part_crops_17"
DEST_ROOT = PROJECT_ROOT / "patchcore_data_17"

PARTS = [
    "front_bumper",
    "rear_bumper",
    "front_door",
    "rear_door",
    "front_window",
    "rear_window",
    "wheel",
    "front_fender",
    "rear_fender",
    "hood",
    "trunk_lid",
    "headlight",
    "taillight",
    "side_mirror",
    "windshield_front",
    "windshield_back",
    "license_plate",
]

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Number of clean images to hold out for testing
TEST_COUNT = 10


print("=" * 80)
print("BUILDING FRESH PATCHCORE DATASET")
print("=" * 80)

if not SOURCE_ROOT.exists():
    raise FileNotFoundError(
        f"Source folder not found:\n{SOURCE_ROOT}"
    )

# Safety check: this script expects the destination to be new/empty.
if DEST_ROOT.exists():
    raise FileExistsError(
        f"\nDestination already exists:\n{DEST_ROOT}\n\n"
        "Delete patchcore_data_17 first, then run this script again."
    )

DEST_ROOT.mkdir(parents=True, exist_ok=True)

grand_total = 0
grand_train = 0
grand_test = 0

print()
print("Source:")
print(SOURCE_ROOT)
print()
print("Destination:")
print(DEST_ROOT)
print()

for part in PARTS:

    source_part = SOURCE_ROOT / part

    if not source_part.exists():
        raise FileNotFoundError(
            f"Missing part folder:\n{source_part}"
        )

    # Find image files
    images = sorted(
        [
            p for p in source_part.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        ],
        key=lambda p: p.name.lower()
    )

    if len(images) == 0:
        raise ValueError(
            f"No images found for part:\n{part}"
        )

    # Make sure the test set is not larger than the dataset
    test_count = min(TEST_COUNT, len(images))

    # First TEST_COUNT images -> test/good
    # Remaining images -> train/good
    test_images = images[:test_count]
    train_images = images[test_count:]

    train_dir = DEST_ROOT / part / "train" / "good"
    test_dir = DEST_ROOT / part / "test" / "good"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy training images
    for src in train_images:
        dst = train_dir / src.name
        shutil.copy2(src, dst)

    # Copy testing images
    for src in test_images:
        dst = test_dir / src.name
        shutil.copy2(src, dst)

    total = len(images)

    grand_total += total
    grand_train += len(train_images)
    grand_test += len(test_images)

    print(
        f"{part:<22} "
        f"total={total:<4} "
        f"train={len(train_images):<4} "
        f"test={len(test_images):<3}"
    )

print()
print("-" * 80)
print(f"{'TOTAL':<22} total={grand_total:<4} "
      f"train={grand_train:<4} test={grand_test:<3}")
print("-" * 80)

print()
print("PATCHCORE DATASET CREATED SUCCESSFULLY")
print()
print(f"Location:")
print(DEST_ROOT)
print()
print("Structure:")
print("patchcore_data_17/")
print(" <part>/")
print(" train/")
print(" good/")
print(" test/")
print(" good/")
print()

print("IMPORTANT:")
print("- All images are clean/undamaged.")
print("- No damaged images are required for this training dataset.")
print("- Each part keeps its own PatchCore dataset.")
print("- Exactly 17 part datasets are created.")
print("=" * 80)