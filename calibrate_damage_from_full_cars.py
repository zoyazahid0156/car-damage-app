from pathlib import Path
import csv
import cv2
import torch

from ultralytics import YOLO

from anomalib.data import PredictDataset
from anomalib.engine import Engine
from anomalib.models import Patchcore
from torch.utils.data import DataLoader


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\zoya.zahid\Documents\car dataset project"
)

# Your 8 full damaged-car images
DAMAGE_IMAGES_DIR = PROJECT_ROOT / "damaged_images"

# Current YOLO detector
YOLO_MODEL = (
    PROJECT_ROOT
    / "detector_results_17"
    / "car_parts_17"
    / "weights"
    / "best.pt"
)

# Production PatchCore models
PATCHCORE_ROOT = PROJECT_ROOT / "patchcore_models_17"

# Where automatically generated damaged crops will be stored
OUTPUT_ROOT = PROJECT_ROOT / "damage_calibration_17"

CROPS_ROOT = OUTPUT_ROOT / "crops"
RESULTS_ROOT = OUTPUT_ROOT / "patchcore_results"

CSV_FILE = OUTPUT_ROOT / "damage_scores_17.csv"


# ============================================================
# FIXED 17 PARTS
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


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

# YOLO confidence.
#
# We use a relatively low confidence because this is a
# calibration experiment and we don't want to throw away
# potentially useful parts.
YOLO_CONFIDENCE = 0.20

YOLO_IOU = 0.50

# PatchCore inference image size.
PATCHCORE_IMAGE_SIZE = (256, 256)

# CPU only
DEVICE = "cpu"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
CROPS_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)


for part in PARTS:
    (CROPS_ROOT / part).mkdir(parents=True, exist_ok=True)


# ============================================================
# CHECK FILES
# ============================================================

print("=" * 70)
print("DAMAGE CALIBRATION FROM FULL-CAR IMAGES")
print("=" * 70)

print("\nChecking files...")

if not DAMAGE_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Damage image folder not found:\n{DAMAGE_IMAGES_DIR}"
    )

if not YOLO_MODEL.exists():
    raise FileNotFoundError(
        f"YOLO model not found:\n{YOLO_MODEL}"
    )

if not PATCHCORE_ROOT.exists():
    raise FileNotFoundError(
        f"PatchCore production folder not found:\n{PATCHCORE_ROOT}"
    )


damage_images = sorted(
    [
        p
        for p in DAMAGE_IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
)


if len(damage_images) == 0:
    raise RuntimeError(
        f"No images found in:\n{DAMAGE_IMAGES_DIR}"
    )


print(f"Damage images found: {len(damage_images)}")

for image_path in damage_images:
    print(" ", image_path.name)


# ============================================================
# LOAD YOLO
# ============================================================

print("\n" + "=" * 70)
print("Loading YOLO detector")
print("=" * 70)

print(f"Model: {YOLO_MODEL}")

yolo_model = YOLO(str(YOLO_MODEL))

print("YOLO loaded successfully.")


# ============================================================
# READ YOLO CLASS NAMES
# ============================================================

print("\nYOLO class names:")

names = yolo_model.names

if isinstance(names, dict):
    yolo_names = names
else:
    yolo_names = {
        i: name
        for i, name in enumerate(names)
    }


for class_id, name in yolo_names.items():
    print(f" {class_id}: {name}")


# ============================================================
# DETECTION + AUTOMATIC CROPPING
# ============================================================

print("\n" + "=" * 70)
print("STEP 1: DETECTING PARTS AND CREATING CROPS")
print("=" * 70)


crop_records = []


for image_number, image_path in enumerate(damage_images, start=1):

    print("\n" + "-" * 70)
    print(
        f"[{image_number}/{len(damage_images)}] "
        f"{image_path.name}"
    )
    print("-" * 70)

    image = cv2.imread(str(image_path))

    if image is None:
        print("WARNING: Could not read image. Skipping.")
        continue

    image_height, image_width = image.shape[:2]

    print(
        f"Image size: "
        f"{image_width} x {image_height}"
    )

    # Run YOLO
    results = yolo_model.predict(
        source=image,
        conf=YOLO_CONFIDENCE,
        iou=YOLO_IOU,
        device=DEVICE,
        verbose=False,
    )

    if not results:
        print("No YOLO result.")
        continue

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        print("No parts detected.")
        continue

    detection_count = 0

    for detection_index, box in enumerate(
        result.boxes,
        start=1
    ):

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        class_name = yolo_names.get(
            class_id,
            str(class_id)
        )

        # Make sure the detected class belongs
        # to our permanent 17-part taxonomy.
        if class_name not in PARTS:
            print(
                f"Skipping unknown class: {class_name}"
            )
            continue

        xyxy = box.xyxy[0].cpu().numpy()

        x1, y1, x2, y2 = [
            int(round(value))
            for value in xyxy
        ]

        # Keep coordinates inside image
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width))
        y2 = max(0, min(y2, image_height))

        if x2 <= x1 or y2 <= y1:
            print(
                f"Invalid box for {class_name}. Skipping."
            )
            continue

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            print(
                f"Empty crop for {class_name}. Skipping."
            )
            continue

        # Create a unique crop filename
        crop_filename = (
            f"{image_path.stem}"
            f"__det{detection_index}"
            f"__{class_name}.jpg"
        )

        crop_path = (
            CROPS_ROOT
            / class_name
            / crop_filename
        )

        cv2.imwrite(
            str(crop_path),
            crop
        )

        detection_count += 1

        crop_records.append(
            {
                "source_image": image_path.name,
                "part": class_name,
                "confidence": confidence,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "crop_path": str(crop_path),
            }
        )

        print(
            f" {class_name:20s} "
            f"confidence={confidence:.3f} "
            f"box=({x1},{y1},{x2},{y2})"
        )

    print(
        f"Parts/crops created: {detection_count}"
    )


# ============================================================
# DETECTION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("DETECTION SUMMARY")
print("=" * 70)

print(
    f"Total automatic crops created: "
    f"{len(crop_records)}"
)


part_crop_counts = {
    part: 0
    for part in PARTS
}


for record in crop_records:
    part_crop_counts[
        record["part"]
    ] += 1


for part in PARTS:
    print(
        f"{part:20s}: "
        f"{part_crop_counts[part]}"
    )


# ============================================================
# SAVE INITIAL CSV
# ============================================================

CSV_FIELDS = [
    "source_image",
    "part",
    "yolo_confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "crop_path",
    "patchcore_score",
]


# ============================================================
# PATCHCORE FUNCTION
# ============================================================

def run_patchcore_for_part(
    part_name,
    crop_paths,
):
    """
    Run one production PatchCore model on all
    automatically generated crops for one part.
    """

    if len(crop_paths) == 0:
        return {}

    model_path = (
        PATCHCORE_ROOT
        / part_name
        / "model.ckpt"
    )

    if not model_path.exists():
        print(
            f"\nWARNING: PatchCore model missing for "
            f"{part_name}"
        )
        print(model_path)
        return {}

    print("\n" + "-" * 70)
    print(
        f"PatchCore: {part_name}"
    )
    print(
        f"Crops: {len(crop_paths)}"
    )
    print(
        f"Model: {model_path}"
    )
    print("-" * 70)

    # PredictDataset is important for Anomalib 2.6.2.
    dataset = PredictDataset(
        path=crop_paths,
        image_size=PATCHCORE_IMAGE_SIZE,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_fn,
    )

    # IMPORTANT:
    # pre_trained=False because the checkpoint already
    # contains the trained PatchCore model.
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=[
            "layer2",
            "layer3",
        ],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
        pre_trained=False,
    )

    engine = Engine(
        default_root_dir=str(RESULTS_ROOT),
        accelerator="cpu",
        devices=1,
    )

    predictions = engine.predict(
        model=model,
        dataloaders=[loader],
        ckpt_path=str(model_path),
        return_predictions=True,
    )

    scores = {}

    if predictions is None:
        print(
            "WARNING: PatchCore returned no predictions."
        )
        return scores

    prediction_list = []

    if isinstance(predictions, list):
        prediction_list = predictions
    else:
        prediction_list = list(predictions)

    for prediction in prediction_list:

        # Anomalib prediction object normally contains
        # pred_score and image_path.
        if not hasattr(
            prediction,
            "pred_score"
        ):
            continue

        raw_score = prediction.pred_score

        if torch.is_tensor(raw_score):
            score = float(
                raw_score.flatten()[0].item()
            )
        else:
            score = float(raw_score)

        image_path = None

        if hasattr(
            prediction,
            "image_path"
        ):
            image_path = prediction.image_path

        if image_path is None:
            continue

        if isinstance(
            image_path,
            (list, tuple)
        ):
            image_path = image_path[0]

        image_path = str(image_path)

        scores[
            str(Path(image_path).resolve())
        ] = score

        print(
            f"{Path(image_path).name:55s}"
            f" score={score:.6f}"
        )

    return scores


# ============================================================
# RUN PATCHCORE FOR ALL PARTS
# ============================================================

print("\n" + "=" * 70)
print("STEP 2: PATCHCORE DAMAGE SCORES")
print("=" * 70)


records_by_part = {
    part: []
    for part in PARTS
}


for record in crop_records:
    records_by_part[
        record["part"]
    ].append(record)


for part in PARTS:

    records = records_by_part[part]

    if len(records) == 0:
        print(
            f"\n{part}: no detected crops. Skipping."
        )
        continue

    crop_paths = [
        Path(record["crop_path"])
        for record in records
    ]

    scores = run_patchcore_for_part(
        part,
        crop_paths,
    )

    for record in records:

        crop_key = str(
            Path(
                record["crop_path"]
            ).resolve()
        )

        record["patchcore_score"] = (
            scores.get(
                crop_key,
                None
            )
        )


# ============================================================
# SAVE FINAL CSV
# ============================================================

print("\n" + "=" * 70)
print("SAVING DAMAGE SCORE REPORT")
print("=" * 70)


with open(
    CSV_FILE,
    "w",
    newline="",
    encoding="utf-8",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=CSV_FIELDS,
    )

    writer.writeheader()

    for record in crop_records:

        writer.writerow(
            {
                "source_image":
                    record["source_image"],

                "part":
                    record["part"],

                "yolo_confidence":
                    f"{record['confidence']:.6f}",

                "x1":
                    record["x1"],

                "y1":
                    record["y1"],

                "x2":
                    record["x2"],

                "y2":
                    record["y2"],

                "crop_path":
                    record["crop_path"],

                "patchcore_score":
                    (
                        ""
                        if record.get(
                            "patchcore_score"
                        ) is None
                        else
                        f"{record['patchcore_score']:.6f}"
                    ),
            }
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)

print(
    f"Full damaged images processed: "
    f"{len(damage_images)}"
)

print(
    f"Automatic crops created: "
    f"{len(crop_records)}"
)

print(
    f"\nCrops saved here:\n"
    f"{CROPS_ROOT}"
)

print(
    f"\nScore report saved here:\n"
    f"{CSV_FILE}"
)

print("\nPart summary:")

for part in PARTS:

    records = records_by_part[part]

    scores = [
        r["patchcore_score"]
        for r in records
        if r.get("patchcore_score") is not None
    ]

    if not scores:
        print(
            f"{part:20s}: no scores"
        )
        continue

    print(
        f"{part:20s}: "
        f"N={len(scores):3d} "
        f"min={min(scores):.6f} "
        f"max={max(scores):.6f} "
        f"avg={sum(scores)/len(scores):.6f}"
    )


print("\nNO DAMAGE THRESHOLDS HAVE BEEN SET YET.")
print(
    "These scores will be used for calibration."
)
print("=" * 70)