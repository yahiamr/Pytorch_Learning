from ultralytics import YOLO

model = YOLO('yolov8s.pt')

#results = model(source=1, show= True, conf = 0.6)

seg_model = YOLO('yolov8n-seg.pt')

results = seg_model(source=1, show= True, conf = 0.6)

