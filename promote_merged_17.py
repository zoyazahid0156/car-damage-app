from pathlib import Path
import json
import shutil

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

TEST_JSON = PROJECT_ROOT / "car_parts_17_coco_merged_test.json"

OLD_MASTER_JSON = PROJECT_ROOT / "car_parts_17_coco.json"

FINAL_JSON = PROJECT_ROOT / "car_parts_17_coco_final.json"

ORIGINAL_IMAGES = PROJECT_ROOT / "car_images"

STAGED_IMAGES = PROJECT_ROOT / "car_images_merged_17"

FINAL_IMAGES = PROJECT_ROOT / "car_images_master_17"


# ============================================================
# START
# ============================================================

print("=" * 80)
print("PROMOTING VALIDATED 17-PART DATASET TO FINAL MASTER")
print("=" * 80)

print()
print("Validated JSON:")
print(TEST_JSON)

print()
print("Original JSON backup:")
print(OLD_MASTER_JSON)

print()
print("Final JSON:")
print(FINAL_JSON)

print()
print("Original images:")
print(ORIGINAL_IMAGES)

print()
print("City staged images:")
print(STAGED_IMAGES)

print()
print("Final image folder:")
print(FINAL_IMAGES)

print()


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not TEST_JSON.exists():
    raise FileNotFoundError(
        f"Validated test JSON not found:\n{TEST_JSON}"
    )

if not OLD_MASTER_JSON.exists():
    raise FileNotFoundError(
        f"Original master JSON not found:\n{OLD_MASTER_JSON}"
    )

if not ORIGINAL_IMAGES.exists():
    raise FileNotFoundError(
        f"Original image folder not found:\n{ORIGINAL_IMAGES}"
    )

if not STAGED_IMAGES.exists():
    raise FileNotFoundError(
        f"Staged City image folder not found:\n{STAGED_IMAGES}"
    )


# ============================================================
# LOAD VALIDATED JSON
# ============================================================

with open(
    TEST_JSON,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


# ============================================================
# BASIC SAFETY CHECK
# ============================================================

if len(data.get("images", [])) != 204:
    raise ValueError(
        "Expected 204 images, but validated JSON contains "
        f"{len(data.get('images', []))}."
    )

if len(data.get("annotations", [])) != 1724:
    raise ValueError(
        "Expected 1724 annotations, but validated JSON contains "
        f"{len(data.get('annotations', []))}."
    )

if len(data.get("categories", [])) != 17:
    raise ValueError(
        "Expected 17 categories, but validated JSON contains "
        f"{len(data.get('categories', []))}."
    )


# ============================================================
# CREATE FINAL IMAGE FOLDER
# ============================================================

if FINAL_IMAGES.exists():

    print("WARNING:")
    print("Final image folder already exists:")
    print(FINAL_IMAGES)
    print()

    answer = input(
        "Type YES to replace it, or anything else to cancel: "
    ).strip()

    if answer != "YES":
        print()
        print("Cancelled.")
        raise SystemExit

    shutil.rmtree(FINAL_IMAGES)


FINAL_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COPY ORIGINAL 41 IMAGES
# ============================================================

print("=" * 80)
print("COPYING ORIGINAL 41 IMAGES")
print("=" * 80)

original_count = 0

for image in data["images"]:

    file_name = Path(
        image["file_name"]
    ).name

    # Original images are the images without
    # teammate_city_data metadata.
    if image.get("source") == "teammate_city_data":
        continue

    source = ORIGINAL_IMAGES / file_name
    destination = FINAL_IMAGES / file_name

    if not source.exists():
        raise FileNotFoundError(
            f"Original image missing:\n{source}"
        )

    shutil.copy2(
        source,
        destination
    )

    original_count += 1


print(
    f"Original images copied: {original_count}"
)


# ============================================================
# COPY CITY IMAGES
# ============================================================

print()
print("=" * 80)
print("COPYING HONDA CITY IMAGES")
print("=" * 80)

city_count = 0

for image in data["images"]:

    if image.get("source") != "teammate_city_data":
        continue

    file_name = Path(
        image["file_name"]
    ).name

    source = STAGED_IMAGES / file_name
    destination = FINAL_IMAGES / file_name

    if not source.exists():
        raise FileNotFoundError(
            f"City staged image missing:\n{source}"
        )

    shutil.copy2(
        source,
        destination
    )

    city_count += 1


print(
    f"Honda City images copied: {city_count}"
)


# ============================================================
# VERIFY FINAL IMAGE COUNT
# ============================================================

print()
print("=" * 80)
print("VERIFYING FINAL IMAGE FOLDER")
print("=" * 80)

final_files = [
    path
    for path in FINAL_IMAGES.iterdir()
    if path.is_file()
]

print(
    "Images in final folder:",
    len(final_files)
)

if len(final_files) != 204:
    raise ValueError(
        "Final image count is incorrect."
    )

print(
    "PASS: Final image folder contains exactly 204 images."
)


# ============================================================
# SAVE FINAL JSON
# ============================================================

print()
print("=" * 80)
print("CREATING FINAL MASTER JSON")
print("=" * 80)

data.setdefault("info", {})

data["info"]["dataset_status"] = "final_master"
data["info"]["total_images"] = 204
data["info"]["total_annotations"] = 1724
data["info"]["total_categories"] = 17

data["info"]["original_master_backup"] = (
    "car_parts_17_coco.json"
)

data["info"]["source_datasets"] = [
    "Original 17-part Honda Civic dataset",
    "Honda City 2016 teammate dataset",
    "Honda City 2017 teammate dataset",
]


with open(
    FINAL_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        data,
        f,
        indent=2
    )


# ============================================================
# VERIFY FINAL JSON
# ============================================================

print()
print("=" * 80)
print("VERIFYING FINAL MASTER JSON")
print("=" * 80)

with open(
    FINAL_JSON,
    "r",
    encoding="utf-8"
) as f:
    final_data = json.load(f)


if len(final_data["images"]) != 204:
    raise ValueError(
        "Final JSON image count is incorrect."
    )

if len(final_data["annotations"]) != 1724:
    raise ValueError(
        "Final JSON annotation count is incorrect."
    )

if len(final_data["categories"]) != 17:
    raise ValueError(
        "Final JSON category count is incorrect."
    )


print("PASS: Final JSON contains 204 images.")
print("PASS: Final JSON contains 1724 annotations.")
print("PASS: Final JSON contains 17 categories.")


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("FINAL MASTER DATASET CREATED")
print("=" * 80)

print()
print("FINAL JSON:")
print(FINAL_JSON)

print()
print("FINAL IMAGE FOLDER:")
print(FINAL_IMAGES)

print()
print("Images:")
print(" Original Honda Civic : 41")
print(" Honda City : 163")
print(" TOTAL : 204")

print()
print("Annotations:")
print(" Original Honda Civic : 820")
print(" Honda City : 904")
print(" TOTAL : 1724")

print()
print("Categories: 17")

print()
print("IMPORTANT:")
print("Your original car_parts_17_coco.json was NOT modified.")
print("Your original car_images folder was NOT modified.")

print()
print("The validated merged dataset is now available as:")
print(FINAL_JSON)

print()
print("=" * 80)
print("PROMOTION COMPLETE")
print("=" * 80)