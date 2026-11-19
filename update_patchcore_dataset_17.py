import csv
import shutil
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

SOURCE_DATASET = PROJECT / "patchcore_data_17_updated"
DEST_DATASET = PROJECT / "patchcore_data_17_new"

CROP_REVIEW_CSV = (
    PROJECT / "part_crops_17" / "new_clean_crops_review.csv"
)

EXPECTED_PARTS = {
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
}


def main():
    # Safety checks: never modify the original dataset.
    if not SOURCE_DATASET.is_dir():
        raise FileNotFoundError(
            f"Original dataset not found: {SOURCE_DATASET}"
        )

    if not CROP_REVIEW_CSV.is_file():
        raise FileNotFoundError(
            f"Crop review CSV not found: {CROP_REVIEW_CSV}"
        )

    if DEST_DATASET.exists():
        raise FileExistsError(
            f"Destination already exists:\n{DEST_DATASET}\n"
            "Nothing was copied or overwritten. Rename or inspect "
            "the existing destination before trying again."
        )

    # Verify the original dataset has exactly the expected part folders.
    actual_parts = {
        p.name for p in SOURCE_DATASET.iterdir() if p.is_dir()
    }

    if actual_parts != EXPECTED_PARTS:
        raise ValueError(
            "The original dataset folders do not match the expected "
            "17 categories.\n"
            f"Missing: {sorted(EXPECTED_PARTS - actual_parts)}\n"
            f"Unexpected: {sorted(actual_parts - EXPECTED_PARTS)}"
        )

    # Read and validate the clean-crop review file before copying.
    rows = []

    with CROP_REVIEW_CSV.open(
        "r", newline="", encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        required_columns = {
            "source_image",
            "part",
            "crop_path",
            "review_status",
        }

        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                "The review CSV is missing required columns: "
                f"{sorted(required_columns - set(reader.fieldnames or []))}"
            )

        for row in reader:
            if row["review_status"].strip().lower() != "clean":
                continue

            part = row["part"].strip()
            if part not in EXPECTED_PARTS:
                raise ValueError(f"Unexpected part category: {part}")

            source_name = Path(row["source_image"].strip()).stem

            # This file should contain only images 42 through 84.
            if not source_name.isdigit() or not (
                42 <= int(source_name) <= 84
            ):
                raise ValueError(
                    f"Unexpected source image in clean-crop CSV: "
                    f"{row['source_image']}"
                )

            crop_path = Path(row["crop_path"].strip())
            if not crop_path.is_absolute():
                crop_path = PROJECT / crop_path

            if not crop_path.is_file():
                raise FileNotFoundError(
                    f"Crop image not found: {crop_path}"
                )

            rows.append({
                "part": part,
                "source_name": source_name,
                "crop_path": crop_path,
            })

    if len(rows) != 776:
        raise ValueError(
            f"Expected 776 clean crops, but found {len(rows)}. "
            "Check the review CSV before continuing."
        )

    # Create a separate copy of the complete original dataset.
    print("Copying the original dataset to a new folder...")
    shutil.copytree(SOURCE_DATASET, DEST_DATASET)

    # Add new crops to train/good only. Leave test/good unchanged.
    added_counts = Counter()

    try:
        for row in rows:
            part = row["part"]
            train_good = DEST_DATASET / part / "train" / "good"

            if not train_good.is_dir():
                raise FileNotFoundError(
                    f"Expected training folder not found: {train_good}"
                )

            crop_stem = row["crop_path"].stem
            destination_name = f"new_{crop_stem}.jpg"
            destination = train_good / destination_name

            if destination.exists():
                raise FileExistsError(
                    f"Refusing to overwrite an image: {destination}"
                )

            shutil.copy2(row["crop_path"], destination)
            added_counts[part] += 1

    except Exception:
        print(
            "\nAn error occurred while adding crops. "
            "The original dataset remains unchanged.\n"
            "The new destination may be incomplete; inspect it before "
            "rerunning the script."
        )
        raise

    # Validate the completed dataset.
    print("\nDATASET UPDATE COMPLETE")
    print(f"Original dataset: {SOURCE_DATASET}")
    print(f"Updated dataset: {DEST_DATASET}")
    print(f"New clean crops added: {sum(added_counts.values())}")

    total_train = 0
    total_test = 0

    print("\nCOUNTS BY PART")
    print(f"{'Part':<22} {'Added':>7} {'Train total':>12} {'Test total':>11}")

    for part in sorted(EXPECTED_PARTS):
        train_good = DEST_DATASET / part / "train" / "good"
        test_good = DEST_DATASET / part / "test" / "good"

        if not train_good.is_dir() or not test_good.is_dir():
            raise FileNotFoundError(
                f"Missing train/good or test/good folder for {part}"
            )

        train_count = sum(
            1 for p in train_good.iterdir()
            if p.is_file()
        )
        test_count = sum(
            1 for p in test_good.iterdir()
            if p.is_file()
        )

        total_train += train_count
        total_test += test_count

        print(
            f"{part:<22} {added_counts[part]:>7} "
            f"{train_count:>12} {test_count:>11}"
        )

    print("-" * 56)
    print(f"{'TOTAL':<22} {sum(added_counts.values()):>7} "
          f"{total_train:>12} {total_test:>11}")

    print("\nThe original patchcore_data_17 folder was not modified.")
    print("No new crops were added to any test/good folder.")
    print("Do not retrain yet; first check the counts above.")


if __name__ == "__main__":
    main()