import json
import csv
from pathlib import Path
from collections import Counter

import cv2

ROOT = Path(".")
JSON_PATH = ROOT / "car_parts_17_coco_final.json"
IMAGE_DIR = ROOT / "car_images"

# New folder: existing part_crops_17 remains untouched.
OUTPUT_DIR = ROOT / "part_crops_17"
CSV_PATH = OUTPUT_DIR / "crop_index.csv"

if OUTPUT_DIR.exists():
    raise FileExistsError(
        f"{OUTPUT_DIR} already exists.\n"
        "Rename it or choose a fresh output folder before running again."
    )

with JSON_PATH.open("r", encoding="utf-8") as f:
    coco = json.load(f)

categories = {
    cat["id"]: cat["name"]
    for cat in coco["categories"]
}

expected_parts = {
    "front_bumper", "rear_bumper", "front_door", "rear_door",
    "front_window", "rear_window", "wheel", "front_fender",
    "quarter_panel", "hood", "trunk_lid", "headlight",
    "taillight", "side_mirror", "windshield_front",
    "windshield_back", "license_plate",
}

if set(categories.values()) != expected_parts:
    raise ValueError("The COCO file does not contain the expected 17 parts.")

images_by_id = {img["id"]: img for img in coco["images"]}

OUTPUT_DIR.mkdir(parents=True)

for part in expected_parts:
    (OUTPUT_DIR / part).mkdir()

rows = []
counts = Counter()
skipped = []

for index, ann in enumerate(coco["annotations"], start=1):
    image_info = images_by_id.get(ann["image_id"])
    if image_info is None:
        skipped.append((ann.get("id"), "unknown image ID"))
        continue

    part = categories.get(ann["category_id"])
    if part is None:
        skipped.append((ann.get("id"), "unknown category ID"))
        continue

    image_name = Path(
        image_info["file_name"].replace("\\", "/")
    ).name
    image_path = IMAGE_DIR / image_name

    image = cv2.imread(str(image_path))
    if image is None:
        skipped.append((ann.get("id"), f"cannot read {image_name}"))
        continue

    height, width = image.shape[:2]
    bbox = ann.get("bbox", [])

    if len(bbox) != 4:
        skipped.append((ann.get("id"), "invalid bbox format"))
        continue

    x, y, box_w, box_h = bbox

    # Convert COCO [x, y, width, height] to integer pixel bounds.
    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(width, int(round(x + box_w)))
    y2 = min(height, int(round(y + box_h)))

    if x2 <= x1 or y2 <= y1:
        skipped.append((ann.get("id"), "empty crop"))
        continue

    crop = image[y1:y2, x1:x2]

    # Include the annotation ID to ensure each crop has a unique name.
    crop_name = (
        f"{Path(image_name).stem}_ann{ann['id']}.jpg"
    )
    crop_path = OUTPUT_DIR / part / crop_name

    if not cv2.imwrite(str(crop_path), crop):
        skipped.append((ann.get("id"), "failed to save crop"))
        continue

    counts[part] += 1
    rows.append({
        "annotation_id": ann["id"],
        "image_id": ann["image_id"],
        "source_image": image_name,
        "part": part,
        "crop_path": str(crop_path),
        "x": x1,
        "y": y1,
        "width": x2 - x1,
        "height": y2 - y1,
    })

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "annotation_id", "image_id", "source_image", "part",
            "crop_path", "x", "y", "width", "height",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print("\nCROP EXTRACTION COMPLETE")
print("Master images:", len(coco["images"]))
print("Master annotations:", len(coco["annotations"]))
print("Crops successfully saved:", len(rows))
print("Skipped annotations:", len(skipped))
print("Output folder:", OUTPUT_DIR)
print("Crop index:", CSV_PATH)

print("\nCrops per part:")
for part in sorted(expected_parts):
    print(f"{part}: {counts[part]}")

if skipped:
    print("\nFirst skipped annotations:")
    for item in skipped[:20]:
        print(item)