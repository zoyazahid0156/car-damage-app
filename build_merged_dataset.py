
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict
import hashlib
import json
import math
import shutil

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
ROOT = Path(r"C:\Users\zoya.zahid\Documents\car dataset project")
ZIP_DIR = ROOT / "datasets_downloaded"
OUT = ROOT / "merged_damage_dataset"

CARDD_ZIP = ZIP_DIR / "CarDD.zip"
DATASET2_ZIP = ZIP_DIR / "Dataset2.zip"

# Canonical YOLO segmentation class order
CLASSES = [
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "lamp_broken",
    "tire_flat",
]

# Dataset2 class IDs from its data.yaml
D2_NAMES = [
    "car-part-crack",
    "flat-tire",
    "glass-crack",
    "lamp-broken",
    "minor-deformation",
    "moderate-deformation",
    "scratch",
    "severe-deformation",
    "side-mirror-crack",
]

# Conservative mapping: ambiguous classes are excluded for now.
D2_MAP = {
    1: 5,  # flat-tire -> tire_flat
    3: 4,  # lamp-broken -> lamp_broken
    6: 1,  # scratch -> scratch
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def valid_polygon_line(line, class_count):
    """Return (class_id, normalized_coordinates) or None."""
    parts = line.split()

    # Segmentation: class ID + at least 3 (x,y) points.
    # Coordinate count must be even.
    if len(parts) < 7 or (len(parts) - 1) % 2:
        return None

    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None

    if not 0 <= class_id < class_count:
        return None

    if not all(math.isfinite(x) and 0 <= x <= 1 for x in coords):
        return None

    return class_id, coords


def write_image_and_label(image_bytes, label_lines, split, source, filename):
    """Write a unique image and matching segmentation label."""
    digest = sha256(image_bytes)

    if digest in seen_hashes:
        report.append(
            f"DUPLICATE_IMAGE_SKIPPED | {source} | {filename} | "
            f"same content as {seen_hashes[digest]}"
        )
        return False

    seen_hashes[digest] = f"{source}/{split}/{filename}"

    stem = Path(filename).stem
    ext = Path(filename).suffix.lower() or ".jpg"
    output_stem = f"{source}_{sha256(image_bytes)[:16]}"

    image_out = OUT / split / "images" / f"{output_stem}{ext}"
    label_out = OUT / split / "labels" / f"{output_stem}.txt"

    image_out.parent.mkdir(parents=True, exist_ok=True)
    label_out.parent.mkdir(parents=True, exist_ok=True)

    image_out.write_bytes(image_bytes)
    label_out.write_text(
        "\n".join(label_lines) + ("\n" if label_lines else ""),
        encoding="utf-8",
    )
    return True


def convert_coco_split(archive, split, json_path, image_folder):
    data = json.loads(archive.read(json_path).decode("utf-8"))

    categories = {
        item["id"]: item["name"].strip().lower()
        for item in data.get("categories", [])
    }

    name_to_class = {
        "dent": 0,
        "scratch": 1,
        "crack": 2,
        "glass shatter": 3,
        "lamp broken": 4,
        "tire flat": 5,
    }

    image_info = {item["id"]: item for item in data.get("images", [])}
    by_image = defaultdict(list)

    for ann in data.get("annotations", []):
        if ann.get("iscrowd", 0):
            report.append(
                f"CARD_CROWD_SKIPPED | annotation={ann.get('id')}"
            )
            continue

        category_name = categories.get(ann.get("category_id"), "")
        class_id = name_to_class.get(category_name)

        if class_id is None:
            report.append(
                f"CARD_UNKNOWN_CLASS_SKIPPED | {category_name}"
            )
            continue

        segmentation = ann.get("segmentation")
        if not isinstance(segmentation, list) or not segmentation:
            report.append(
                f"CARD_NO_POLYGON_SKIPPED | annotation={ann.get('id')}"
            )
            continue

        # COCO polygon format can contain multiple polygons.
        # Keep the largest polygon to avoid incorrectly joining
        # separate disconnected regions into one outline.
        polygons = [
            p for p in segmentation
            if isinstance(p, list) and len(p) >= 6 and len(p) % 2 == 0
        ]
        if not polygons:
            report.append(
                f"CARD_BAD_POLYGON_SKIPPED | annotation={ann.get('id')}"
            )
            continue

        polygon = max(polygons, key=len)
        width = image_info[ann["image_id"]]["width"]
        height = image_info[ann["image_id"]]["height"]

        coords = []
        good = True
        for i in range(0, len(polygon), 2):
            x, y = polygon[i], polygon[i + 1]
            if not (math.isfinite(x) and math.isfinite(y)):
                good = False
                break
            if width <= 0 or height <= 0:
                good = False
                break
            coords.extend([x / width, y / height])

        if not good or not all(0 <= v <= 1 for v in coords):
            report.append(
                f"CARD_INVALID_COORDS_SKIPPED | annotation={ann.get('id')}"
            )
            continue

        line = str(class_id) + " " + " ".join(
            f"{v:.6f}" for v in coords
        )
        by_image[ann["image_id"]].append(line)

    for image_id, info in image_info.items():
        rel_name = info["file_name"]
        image_path = f"{image_folder}/{rel_name}"

        if image_path not in archive.namelist():
            # Some COCO files store a relative path already.
            if rel_name in archive.namelist():
                image_path = rel_name
            else:
                report.append(f"CARD_IMAGE_MISSING | {image_path}")
                continue

        lines = by_image.get(image_id, [])
        # Empty labels are allowed: image has no retained target class.
        write_image_and_label(
            archive.read(image_path),
            lines,
            split,
            "cardd",
            Path(rel_name).name,
        )


def convert_dataset2(archive):
    names = archive.namelist()
    image_names = [
        n for n in names
        if "/images/" in n.replace("\\", "/").lower()
        and Path(n).suffix.lower() in IMAGE_EXTS
    ]

    label_lookup = {}
    for name in names:
        norm = name.replace("\\", "/")
        if "/labels/" in norm.lower() and norm.lower().endswith(".txt"):
            label_lookup[Path(name).stem] = name

    # Preserve Dataset2's provided splits.
    for image_name in image_names:
        parts = image_name.replace("\\", "/").split("/")
        split = next(
            (p for p in parts if p.lower() in {"train", "valid", "test"}),
            None,
        )
        if split is None:
            report.append(f"D2_UNKNOWN_SPLIT | {image_name}")
            continue

        split = {"valid": "val"}.get(split.lower(), split.lower())
        label_name = label_lookup.get(Path(image_name).stem)

        if label_name is None:
            report.append(f"D2_LABEL_MISSING | {image_name}")
            continue

        raw = archive.read(label_name).decode(
            "utf-8", errors="replace"
        ).strip()

        converted = []
        for line_number, line in enumerate(raw.splitlines(), start=1):
            parsed = valid_polygon_line(line, len(D2_NAMES))
            if parsed is None:
                report.append(
                    f"D2_INVALID_LINE_SKIPPED | {label_name} | "
                    f"line={line_number} | {line[:180]}"
                )
                continue

            class_id, coords = parsed

            # Ignore classes whose meanings are not yet verified.
            if class_id not in D2_MAP:
                report.append(
                    f"D2_CLASS_EXCLUDED | {D2_NAMES[class_id]} | "
                    f"{label_name} | line={line_number}"
                )
                continue

            target_id = D2_MAP[class_id]
            converted.append(
                str(target_id) + " " + " ".join(
                    f"{v:.6f}" for v in coords
                )
            )

                # Skip images with no retained damage annotations.
        # Their original damage may belong to excluded classes.
        if not converted:
            report.append(
                f"D2_IMAGE_SKIPPED_NO_RETAINED_CLASSES | {label_name}"
            )
            continue

        write_image_and_label(
            archive.read(image_name),
            converted,
            split,
            "dataset2",
            Path(image_name).name,
        )


# -------------------------------------------------------
# BUILD
# -------------------------------------------------------
if not CARDD_ZIP.exists() or not DATASET2_ZIP.exists():
    raise FileNotFoundError(
        "Check that CarDD.zip and Dataset2.zip are in datasets_downloaded."
    )

if OUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUT}\n"
        "Rename or move it before running again; nothing was overwritten."
    )

report = []
seen_hashes = {}

for split in ("train", "val", "test"):
    (OUT / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT / split / "labels").mkdir(parents=True, exist_ok=True)

try:
    with ZipFile(CARDD_ZIP) as archive:
        base = "CarDD_release/CarDD_COCO"
        for split, json_split, image_folder in [
            ("train", "train2017", f"{base}/train2017"),
            ("val", "val2017", f"{base}/val2017"),
            ("test", "test2017", f"{base}/test2017"),
        ]:
            json_path = f"{base}/annotations/instances_{json_split}.json"
            print("Converting CarDD:", split)
            convert_coco_split(
                archive, split, json_path, image_folder
            )

    with ZipFile(DATASET2_ZIP) as archive:
        print("Converting compatible Dataset2 annotations...")
        convert_dataset2(archive)

    yaml_text = (
        "path: " + str(OUT).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "names:\n"
    )
    for i, name in enumerate(CLASSES):
        yaml_text += f"  {i}: {name}\n"

    (OUT / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (OUT / "merge_report.txt").write_text(
        "\n".join(report) + ("\n" if report else ""),
        encoding="utf-8",
    )

    print("\nMERGE COMPLETE")
    print("Output:", OUT)
    print("Config:", OUT / "data.yaml")
    print("Report:", OUT / "merge_report.txt")
    print("Report entries:", len(report))
    print("Original ZIP files were not modified.")

except Exception:
    # Keep any partial output for inspection; do not delete user files.
    print("\nThe build stopped because of an error.")
    print("Partial output, if any, is at:", OUT)
    raise