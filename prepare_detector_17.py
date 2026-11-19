from pathlib import Path
import json
import shutil
import random


# ============================================================
# SETTINGS
# ============================================================

COCO_FILE = Path("car_parts_17_coco_final.json")
IMAGE_ROOT = Path("car_images")

OUTPUT_ROOT = Path("detector_dataset_17")

TRAIN_RATIO = 0.8
SEED = 42


# ============================================================
# 17 FIXED PARTS
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
# LOAD COCO
# ============================================================

print("=" * 80)
print("PREPARING 17-CLASS DETECTOR DATASET")
print("=" * 80)

if not COCO_FILE.exists():
    raise FileNotFoundError(
        f"COCO file not found: {COCO_FILE.resolve()}"
    )

if not IMAGE_ROOT.exists():
    raise FileNotFoundError(
        f"Image folder not found: {IMAGE_ROOT.resolve()}"
    )

with open(COCO_FILE, "r", encoding="utf-8") as f:
    coco = json.load(f)


# ============================================================
# CATEGORY CHECK
# ============================================================

categories = sorted(
    coco["categories"],
    key=lambda x: x["id"]
)

category_id_to_name = {
    c["id"]: c["name"]
    for c in categories
}

print()
print(f"COCO images : {len(coco['images'])}")
print(f"COCO annotations : {len(coco['annotations'])}")
print(f"COCO categories : {len(categories)}")


if len(categories) != 17:
    raise RuntimeError(
        f"Expected exactly 17 categories, found {len(categories)}"
    )


if set(category_id_to_name.values()) != set(PARTS):
    print()
    print("❌ CATEGORY MISMATCH")
    print("Found categories:")
    for name in category_id_to_name.values():
        print(f" {name}")

    raise RuntimeError(
        "COCO categories do not exactly match the 17 fixed parts."
    )


print("✅ Category check passed.")


# ============================================================
# CREATE YOLO CLASS MAP
# ============================================================

# YOLO requires class IDs starting from 0.
#
# We use the order in PARTS as the permanent class order.

class_id = {
    name: index
    for index, name in enumerate(PARTS)
}


# ============================================================
# PREPARE OUTPUT FOLDERS
# ============================================================

if OUTPUT_ROOT.exists():
    print()
    print("Removing previous detector dataset...")

    shutil.rmtree(OUTPUT_ROOT)

for split in ["train", "val"]:
    (OUTPUT_ROOT / "images" / split).mkdir(
        parents=True,
        exist_ok=True
    )

    (OUTPUT_ROOT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# IMAGE LOOKUP
# ============================================================

images_by_id = {
    image["id"]: image
    for image in coco["images"]
}


annotations_by_image = {}

for annotation in coco["annotations"]:

    image_id = annotation["image_id"]

    annotations_by_image.setdefault(
        image_id,
        []
    ).append(annotation)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

image_ids = list(images_by_id.keys())

random.seed(SEED)
random.shuffle(image_ids)

split_index = int(
    len(image_ids) * TRAIN_RATIO
)

train_ids = image_ids[:split_index]
val_ids = image_ids[split_index:]


print()
print(f"Training images : {len(train_ids)}")
print(f"Validation images : {len(val_ids)}")


# ============================================================
# COCO BOX → YOLO BOX
# ============================================================

def coco_to_yolo(
    bbox,
    image_width,
    image_height
):
    x, y, width, height = bbox

    x_center = x + width / 2
    y_center = y + height / 2

    x_center /= image_width
    y_center /= image_height
    width /= image_width
    height /= image_height

    return (
        x_center,
        y_center,
        width,
        height
    )


# ============================================================
# PROCESS IMAGES
# ============================================================

def process_split(image_ids, split):

    copied = 0
    annotation_count = 0

    for image_id in image_ids:

        image_info = images_by_id[image_id]

        file_name = image_info["file_name"]

        source_image = IMAGE_ROOT / file_name

        if not source_image.exists():

            # Try only the basename as a fallback.
            source_image = IMAGE_ROOT / Path(file_name).name

        if not source_image.exists():

            raise FileNotFoundError(
                f"Image not found: {file_name}"
            )

        output_image = (
            OUTPUT_ROOT
            / "images"
            / split
            / source_image.name
        )

        output_label = (
            OUTPUT_ROOT
            / "labels"
            / split
            / f"{source_image.stem}.txt"
        )

        shutil.copy2(
            source_image,
            output_image
        )

        lines = []

        annotations = annotations_by_image.get(
            image_id,
            []
        )

        for annotation in annotations:

            category_name = category_id_to_name[
                annotation["category_id"]
            ]

            if category_name not in class_id:
                continue

            bbox = annotation["bbox"]

            x_center, y_center, width, height = (
                coco_to_yolo(
                    bbox,
                    image_info["width"],
                    image_info["height"]
                )
            )

            # Clamp values to valid YOLO range.
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            line = (
                f"{class_id[category_name]} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

            lines.append(line)

            annotation_count += 1

        with open(
            output_label,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(lines)
            )

        copied += 1

    return copied, annotation_count


# ============================================================
# PROCESS TRAIN
# ============================================================

train_images, train_annotations = process_split(
    train_ids,
    "train"
)


# ============================================================
# PROCESS VALIDATION
# ============================================================

val_images, val_annotations = process_split(
    val_ids,
    "val"
)


# ============================================================
# CREATE DATA.YAML
# ============================================================

yaml_file = OUTPUT_ROOT / "data.yaml"

yaml_lines = [
    f"path: {OUTPUT_ROOT.resolve()}",
    "train: images/train",
    "val: images/val",
    "",
    "names:",
]

for index, name in enumerate(PARTS):

    yaml_lines.append(
        f" {index}: {name}"
    )


with open(
    yaml_file,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(yaml_lines)
    )


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 80)
print("DATASET PREPARATION COMPLETE")
print("=" * 80)

print()
print(f"Training images : {train_images}")
print(f"Training annotations : {train_annotations}")

print()
print(f"Validation images : {val_images}")
print(f"Validation annotations : {val_annotations}")

print()
print(f"Total images : {train_images + val_images}")
print(f"Total annotations : {train_annotations + val_annotations}")

print()
print("Output:")
print(OUTPUT_ROOT.resolve())

print()
print("data.yaml:")
print(yaml_file.resolve())

print()
print("17 classes:")

for index, name in enumerate(PARTS):
    print(f" {index:2d} -> {name}")

print()
print("✅ Detector dataset ready.")

print("=" * 80)