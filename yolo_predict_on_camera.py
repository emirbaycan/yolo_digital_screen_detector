import os
import cv2
import av
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt

# RTSP Stream URL
username = "admin"
password = "password"
ip = "192.168.1.98"
port = "554"
rtsp_url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/onvif1"

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train10/weights/best.pt")  

# Output directory to save images
output_folder = "camera_images"
os.makedirs(output_folder, exist_ok=True)

# Open the RTSP stream
container = av.open(rtsp_url)
stream = container.streams.video[0]
stream_time_base = stream.time_base
last_saved_time = 0
save_interval = 1  # Save one frame every second

# Open a window for real-time display
cv2.namedWindow("YOLO Live Detection", cv2.WINDOW_NORMAL)
save = False
# Process frames from the RTSP stream
for frame in container.decode(video=0):
    if frame.pts is None:
        continue

    frame_time = frame.pts * stream_time_base

    # Convert frame to a numpy array (BGR format for OpenCV)
    img = frame.to_ndarray(format='bgr24')

    # Run YOLO inference on the frame
    results = model(img, conf=0.5)  # Adjust confidence threshold if needed

    # Annotate the frame with detected objects
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            label = "screen"  # We only have one class

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the annotated frame
    cv2.imshow("YOLO Live Detection", img)

    
    # Save frame only if the specified interval has passed
    if save and frame_time - last_saved_time >= save_interval:
        last_saved_time = frame_time
        timestamp = int(time.time())
        filename = os.path.join(output_folder, f"frame_{timestamp}.png")
        cv2.imwrite(filename, img)
        print(f"Saved: {filename}")
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
container.close()