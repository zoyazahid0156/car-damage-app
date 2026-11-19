from pathlib import Path
import json
import shutil
import random

PROJECT = Path.cwd()

COCO_FILE = PROJECT / "car_parts_17_coco_final.json"
IMAGE_FOLDER = PROJECT / "car_images"

OUTPUT = PROJECT / "yolo_dataset_17"

TRAIN_RATIO = 0.8

# ---------------------------------------------------
# Load COCO
# ---------------------------------------------------

with open(COCO_FILE, "r") as f:
    coco = json.load(f)

# ---------------------------------------------------
# Categories
# ---------------------------------------------------

categories = sorted(
    coco["categories"],
    key=lambda x: x["id"]
)

cat_map = {}

for i, cat in enumerate(categories):
    cat_map[cat["id"]] = i

print("\nClasses:\n")

for i, cat in enumerate(categories):
    print(i, cat["name"])

# ---------------------------------------------------
# Images
# ---------------------------------------------------

images = {}

for img in coco["images"]:
    images[img["id"]] = img

# ---------------------------------------------------
# Annotations
# ---------------------------------------------------

ann_dict = {}

for ann in coco["annotations"]:

    image_id = ann["image_id"]

    if image_id not in ann_dict:
        ann_dict[image_id] = []

    ann_dict[image_id].append(ann)

# ---------------------------------------------------
# Split
# ---------------------------------------------------

image_ids = list(images.keys())

random.seed(42)
random.shuffle(image_ids)

train_n = int(len(image_ids) * TRAIN_RATIO)

train_ids = image_ids[:train_n]
val_ids = image_ids[train_n:]

print()
print("Train:", len(train_ids))
print("Val:", len(val_ids))

# ---------------------------------------------------
# Folders
# ---------------------------------------------------

for split in ["train", "val"]:

    (OUTPUT / "images" / split).mkdir(
        parents=True,
        exist_ok=True,
    )

    (OUTPUT / "labels" / split).mkdir(
        parents=True,
        exist_ok=True,
    )

# ---------------------------------------------------
# Convert
# ---------------------------------------------------

def process(ids, split):

    for image_id in ids:

        img = images[image_id]

        file_name = img["file_name"]

        width = img["width"]
        height = img["height"]

        src = IMAGE_FOLDER / file_name

        dst = OUTPUT / "images" / split / file_name

        shutil.copy(src, dst)

        label_file = (
            OUTPUT
            / "labels"
            / split
            / f"{Path(file_name).stem}.txt"
        )

        lines = []

        for ann in ann_dict.get(image_id, []):

            cls = cat_map[
                ann["category_id"]
            ]

            x, y, w, h = ann["bbox"]

            xc = (x + w/2) / width
            yc = (y + h/2) / height
            w /= width
            h /= height

            lines.append(
                f"{cls} {xc} {yc} {w} {h}"
            )

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

process(train_ids, "train")
process(val_ids, "val")

print()
print("YOLO dataset created:")
print(OUTPUT)