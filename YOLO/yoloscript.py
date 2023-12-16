from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model(source=1, show= True, conf = 0.6)

