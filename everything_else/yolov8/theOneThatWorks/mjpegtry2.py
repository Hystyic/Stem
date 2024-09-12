import cv2
import numpy as np
import urllib.request
from ultralytics import YOLO
import time

# Initialize YOLOv8 model
model = YOLO("yolov8n.pt")

# URL of the MJPEG stream
stream_url = 'http://192.168.0.207/640x480.mjpeg'

# Function to fetch a frame from the MJPEG stream
def fetch_frame(stream_url):
    while True:
        try:
            with urllib.request.urlopen(stream_url) as stream:
                bytes = b''
                while True:
                    bytes += stream.read(1024)
                    a = bytes.find(b'\xff\xd8')  # Start of JPEG
                    b = bytes.find(b'\xff\xd9')  # End of JPEG
                    if a != -1 and b != -1:
                        jpg = bytes[a:b+2]
                        bytes = bytes[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        return frame
        except ConnectionResetError:
            print("Connection was reset. Reconnecting...")
            time.sleep(1) 
            
while True:
    frame = fetch_frame(stream_url)
    if frame is not None:
        # Object detection
        results = model(frame)

        # Debug: Print raw results
        print(f"Raw results: {results}")

        # Iterate over the detections
        for det in results:
            if len(det) >= 6:
                xmin, ymin, xmax, ymax, conf, cls_id = map(int, det[:6])
                if conf > 0.1:  # Lower the threshold to 0.1 for testing
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    label = f"{model.names[cls_id]} {conf:.2f}"
                    cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print(f"Detection: {label}")

        # Display the frame
        cv2.imshow("YOLOv8n Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
