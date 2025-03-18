from ultralytics import YOLO

model = YOLO('yolo11n-pose.pt')

results = model(source='fall.mp4', show=True, conf=0.3, save=False)