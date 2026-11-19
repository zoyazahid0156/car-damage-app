from ultralytics import YOLO

model = YOLO("yolo26s.pt")  # load a pretrained model (recommended for training)

model.train(
    data="yolo_dataset_17/dataset.yaml",
    epochs=100,
    imgsz=960,
    batch=8,
    workers=2,
    device="cpu",
    project="detector_results_17",
    name="car_parts_635"
)