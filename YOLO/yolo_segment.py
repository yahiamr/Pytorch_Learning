from ultralytics import YOLO

seg_model = YOLO('yolov8n-seg.pt')

results = seg_model(source=1, show= True, conf = 0.6)

