from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train10/weights/best.pt")  

# Run inference on a test image
results = model("camera_images/frame_1738193401.png", conf=0.5)

for result in results:
    result.save(filename="output.png")  # Save the output image