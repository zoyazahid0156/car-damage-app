
from pathlib import Path
from ultralytics import YOLO

# CHANGE THESE TWO PATHS
MODEL_PATH = r"damage_results\damage_seg\weights\best.pt"
VIDEO_PATH = r"accident.mp4"

model = YOLO(MODEL_PATH)

if model.task != "segment":
    raise RuntimeError(
        f"Expected a segmentation model, got task={model.task!r}"
    )

video = Path(VIDEO_PATH)
if not video.exists():
    raise FileNotFoundError(f"Video not found: {video}")

# Test one setting at a time and compare the output videos.
tests = [
    {"name": "baseline", "imgsz": 640, "conf": 0.25},
    {"name": "lower_conf", "imgsz": 640, "conf": 0.10},
    {"name": "larger_image", "imgsz": 960, "conf": 0.10},
]

for test in tests:
    print(f"Running test: {test['name']}")

    model.predict(
        source=str(video),
        imgsz=test["imgsz"],
        conf=test["conf"],
        device="cpu",           # Use 0 for NVIDIA GPU
        save=True,
        show=False,
        project="runs/video_tests",
        name=test["name"],
        exist_ok=True,
        verbose=False,
    )

print("Tests complete. Check runs/video_tests/")
