from ultralytics import YOLO

model = YOLO("runs/detect/detector_results_17/car_parts_635/weights/last.pt")
model.train(resume=True)