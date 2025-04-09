import os
import cv2
import av
import time

# RTSP Stream URL
username = "admin"
password = "password"
ip = "192.168.1.98"
port = "554"
rtsp_url = "rtsp://" + username + ":" + password + "@" + ip + ":" + port + "/onvif1"


# Output directory to save images
output_folder = "camera_images"
os.makedirs(output_folder, exist_ok=True)

# Capture and save images from the camera
def capture_images_from_camera():
    container = av.open(rtsp_url)
    stream = container.streams.video[0]
    stream_time_base = stream.time_base
    last_saved_time = 0
    save_interval = 1  # Save one frame every second

    for frame in container.decode(video=0):
        # Ensure frame.pts is valid
        if frame.pts is None:
            continue

        # Calculate the timestamp of the current frame
        frame_time = frame.pts * stream_time_base

        # Save frame only if the specified interval has passed
        if frame_time - last_saved_time >= save_interval:
            last_saved_time = frame_time

            # Convert frame to a numpy array
            img = frame.to_ndarray(format='bgr24')

            # Save image in RGB format
            timestamp = int(time.time())
            filename = os.path.join(output_folder, f"frame_{timestamp}.png")
            cv2.imwrite(filename, img)
            print(f"Saved: {filename}")

# Run the function
capture_images_from_camera()