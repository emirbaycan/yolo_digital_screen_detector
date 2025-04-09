from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Load a pre-trained model
model.train(data='data.yaml', epochs=50, imgsz=640, rect=True)