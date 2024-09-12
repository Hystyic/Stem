import cv2
import urllib.request
import numpy as np
from pathlib import Path

# The URL of the ESP32CAM web server
ESP32CAM_URL = "http://192.168.0.208/cam-hi.jpg"

# Import YOLOv5 from Ultralytics
import sys
sys.path.append("path/to/yolov5")  # replace with the path to the yolov5 repository
from yolov5.infer import detect

# Capture a frame from the ESP32CAM web server
def capture_frame():
    img_resp = urllib.request.urlopen(ESP32CAM_URL)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)
    return frame

# Perform object detection using Ultralytics YOLOv5
def detect_objects(frame):
    results = detect(frame, source="inference/images", weights="yolov5s.pt")
    return results.imgs[0]

# Display a frame using OpenCV
def display_frame(frame):
    cv2.imshow("ESP32CAM Feed", frame)

# Start the live feed with Ultralytics YOLOv5 object detection
def live_feed():
    while True:
        frame = capture_frame()
        frame_with_objects = detect_objects(frame)
        display_frame(frame_with_objects)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

# Run the live feed
live_feed()