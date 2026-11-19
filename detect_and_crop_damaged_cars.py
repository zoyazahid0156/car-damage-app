from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

PROJECT = Path(
    r"C:\Users\zoya.zahid\Documents\car dataset project"
)

MODEL_PATH = (
    PROJECT / "runs" / "detect" / "detector_results_17"
    / "car_parts_new" / "weights" / "best.pt"
)

INPUT_DIR = PROJECT / "damaged_images"
OUTPUT_DIR = PROJECT / "damage_calibration_17_updated"
CROPS_DIR = OUTPUT_DIR / "crops"
ANNOTATED_DIR = OUTPUT_DIR / "annotated_images"
CSV_PATH = OUTPUT_DIR / "detected_parts.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONFIDENCE = 0.20
IMAGE_SIZE = 640

PARTS = [
    "front_bumper", "rear_bumper", "front_door", "rear_door",
    "front_window", "rear_window", "wheel", "front_fender",
    "rear_fender", "hood", "trunk_lid", "headlight", "taillight",
    "side_mirror", "windshield_front", "windshield_back",
    "license_plate",
]

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"YOLO model not found: {MODEL_PATH}")
if not INPUT_DIR.exists():
    raise FileNotFoundError(f"Input folder not found: {INPUT_DIR}")

images = sorted(
    p for p in INPUT_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
)
if not images:
    raise RuntimeError(f"No images found in {INPUT_DIR}")

for part in PARTS:
    (CROPS_DIR / part).mkdir(parents=True, exist_ok=True)
ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model = YOLO(str(MODEL_PATH))
class_names = model.names

def class_name_for(class_id):
    if isinstance(class_names, dict):
        return class_names.get(class_id, str(class_id))
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return str(class_id)

rows = []
print(f"Images to process: {len(images)}")

for image_path in images:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"WARNING: Could not read {image_path.name}")
        continue

    height, width = image.shape[:2]
    result = model.predict(
        source=image,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )[0]

    annotated = image.copy()
    count = 0

    if result.boxes is not None:
        for index, box in enumerate(result.boxes, start=1):
            class_id = int(box.cls[0].item())
            part = class_name_for(class_id)

            if part not in PARTS:
                print(f"Skipping unexpected class: {part}")
                continue

            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = [
                int(round(v)) for v in box.xyxy[0].cpu().tolist()
            ]

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width))
            y2 = max(0, min(y2, height))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_name = (
                f"{image_path.stem}__det{index}__{part}.jpg"
            )
            crop_path = CROPS_DIR / part / crop_name

            if not cv2.imwrite(str(crop_path), crop):
                print(f"WARNING: Could not save {crop_path}")
                continue

            cv2.rectangle(
                annotated, (x1, y1), (x2, y2), (0, 255, 0), 2
            )
            label = f"{part} {confidence:.2f}"
            cv2.putText(
                annotated, label, (x1, max(20, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

            rows.append({
                "source_image": image_path.name,
                "part": part,
                "confidence": round(confidence, 6),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "crop_path": str(crop_path),
            })
            count += 1

    annotated_path = ANNOTATED_DIR / f"{image_path.stem}_detected.jpg"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"{image_path.name}: {count} parts detected")

with open(CSV_PATH, "w", newline="", encoding="utf-8") as file:
    columns = [
        "source_image", "part", "confidence",
        "x1", "y1", "x2", "y2", "crop_path"
    ]
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("\nFinished.")
print(f"Total detected crops: {len(rows)}")
print(f"Annotated images: {ANNOTATED_DIR}")
print(f"Part crops: {CROPS_DIR}")
print(f"Detection report: {CSV_PATH}")
print("Review the boxes and crops before using their scores.")