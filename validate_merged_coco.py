import json
from pathlib import Path
from collections import Counter

json_path = Path("car_parts_17_coco_final.json")
image_dir = Path("car_images")

EXPECTED_PARTS = {
    "front_bumper", "rear_bumper", "front_door", "rear_door",
    "front_window", "rear_window", "wheel", "quarter_panel",
    "rear_fender", "hood", "trunk_lid", "headlight",
    "taillight", "side_mirror", "windshield_front",
    "windshield_back", "license_plate",
}

with json_path.open("r", encoding="utf-8") as f:
    data = json.load(f)

images = data["images"]
annotations = data["annotations"]
categories = data["categories"]

print("Images:", len(images))
print("Annotations:", len(annotations))
print("Categories:", len(categories))

image_ids = [img["id"] for img in images]
annotation_ids = [ann["id"] for ann in annotations]
category_ids = [cat["id"] for cat in categories]

print("Duplicate image IDs:", len(image_ids) - len(set(image_ids)))
print("Duplicate annotation IDs:",
      len(annotation_ids) - len(set(annotation_ids)))

category_names = {cat["name"] for cat in categories}
print("All 17 categories correct:", category_names == EXPECTED_PARTS)
print("Category names:", sorted(category_names))

known_image_ids = set(image_ids)
known_category_ids = set(category_ids)

bad_image_refs = [
    ann for ann in annotations
    if ann["image_id"] not in known_image_ids
]
bad_category_refs = [
    ann for ann in annotations
    if ann["category_id"] not in known_category_ids
]

print("Invalid image references:", len(bad_image_refs))
print("Invalid category references:", len(bad_category_refs))

filenames = [
    Path(img["file_name"].replace("\\", "/")).name
    for img in images
]
print("Duplicate filenames:", len(filenames) - len(set(filenames)))

missing_files = [
    name for name in filenames
    if not (image_dir / name).is_file()
]
print("Missing image files:", len(missing_files))
if missing_files:
    print("Examples:", missing_files[:10])

image_dimensions = {
    img["id"]: (img.get("width"), img.get("height"))
    for img in images
}

bad_boxes = []
for ann in annotations:
    bbox = ann.get("bbox")
    dimensions = image_dimensions.get(ann["image_id"])

    if not isinstance(bbox, list) or len(bbox) != 4 or not dimensions:
        bad_boxes.append(ann.get("id"))
        continue

    x, y, w, h = bbox
    width, height = dimensions

    if (
        not all(isinstance(v, (int, float)) for v in bbox)
        or width is None or height is None
        or w <= 0 or h <= 0
        or x < 0 or y < 0
        or x + w > width + 1
        or y + h > height + 1
    ):
        bad_boxes.append(ann.get("id"))

print("Invalid bounding boxes:", len(bad_boxes))

counts = Counter(ann["category_id"] for ann in annotations)
print("\nAnnotations per part:")
for cat in categories:
    print(f"{cat['name']}: {counts.get(cat['id'], 0)}")

print("\nValidation finished. No files were modified.")