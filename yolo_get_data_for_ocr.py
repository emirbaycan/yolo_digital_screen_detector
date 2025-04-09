import os
import cv2
import av
import time
from ultralytics import YOLO

# RTSP Stream URL
username = "admin"
password = "password"
ip = "192.168.1.98"
port = "554"
rtsp_url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/onvif1"

# Load the trained YOLOv8 model
model = YOLO("runs/detect/train10/weights/best.pt")

# Output directories
output_folder = "camera_images"  # Full frames
cropped_screens_folder = "cropped_screens"  # Cropped screen regions
os.makedirs(output_folder, exist_ok=True)
os.makedirs(cropped_screens_folder, exist_ok=True)

# Open the RTSP stream
container = av.open(rtsp_url)
stream = container.streams.video[0]
stream_time_base = stream.time_base
last_saved_time = 0
save_interval = 1  # Save at most 2 images per second

# Open a window for real-time display
cv2.namedWindow("YOLO Live Detection", cv2.WINDOW_NORMAL)
save = True  # Set to True to enable saving images

# Process frames from the RTSP stream
for frame in container.decode(video=0):
    if frame.pts is None:
        continue

    frame_time = frame.pts * stream_time_base
    timestamp = int(time.time())

    # Convert frame to numpy array (BGR format for OpenCV)
    img = frame.to_ndarray(format="bgr24")

    # Run YOLO inference on the frame
    results = model(img, conf=0.5)  # Adjust confidence threshold if needed

    # Track if we have saved a cropped image in this second
    cropped_saved = False

    # Annotate and save the detected screen regions
    for result in results:
        for i, box in enumerate(result.boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            confidence = box.conf[0]  # Confidence score
            label = "screen"  # We only have one class

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                img,
                f"{label} {confidence:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # **Crop the detected screen region correctly**
            cropped_screen = img[y1:y2, x1:x2]

            # Ensure the cropped image is valid (non-empty)
            if (
                cropped_screen.shape[0] > 0
                and cropped_screen.shape[1] > 0
                and not cropped_saved
            ):
                # Save cropped screen region (only one per second)
                screen_filename = os.path.join(
                    cropped_screens_folder, f"screen_{timestamp}.png"
                )
                cv2.imwrite(screen_filename, cropped_screen)
                print(f"Saved cropped screen: {screen_filename}")
                cropped_saved = (
                    True  # Prevent saving multiple cropped images in the same second
                )

    # Display the annotated frame
    cv2.imshow("YOLO Live Detection", img)

    # Save full frame only if the interval has passed
    if save and frame_time - last_saved_time >= save_interval:
        last_saved_time = frame_time
        frame_filename = os.path.join(output_folder, f"frame_{timestamp}.png")
        cv2.imwrite(frame_filename, img)
        print(f"Saved full frame: {frame_filename}")

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
container.close()
