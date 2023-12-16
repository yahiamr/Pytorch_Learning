from ultralytics import YOLO

model = YOLO('yolov8s.pt')

results = model(source='./road.jpg', show= True, conf = 0.4,save=True)

