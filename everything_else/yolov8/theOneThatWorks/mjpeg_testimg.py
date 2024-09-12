import cv2
import math
import urllib.request
from ultralytics import YOLO
import numpy as np
import time

# Initialize YOLOv8 model
model = YOLO("yolo-Weights/yolov8n.pt")

# URLs of the MJPEG streams
stream_url1 = 'http://192.168.137.175/640x480.mjpeg'
stream_url2 = 'http://192.168.137.175/640x480.mjpeg'

# Function to fetch a frame from the MJPEG stream
def fetch_frame(stream_url):
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
                    frame = cv2.imdecode(
                        np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    return frame
    except ConnectionResetError:
        print("Connection was reset. Reconnecting...")
        time.sleep(1)

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
import winsound

def get_sound_params(box):
    # Calculate the area of the bounding box
    area = (box[2] - box[0]) * (box[3] - box[1])
    
    # Map the area to a frequency and rate
    # These thresholds and values can be adjusted based on your specific requirements
    if area > 50000:      # Very close object
        frequency = 500  # Lower frequency
        rate = 10        # Faster beeping
    elif area > 20000:    # Moderately close object
        frequency = 1000
        rate = 5
    else:                 # Farther object
        frequency = 1500
        rate = 2

    return frequency, rate
while True:
    frame1 = fetch_frame(stream_url1)
    frame2 = fetch_frame(stream_url2)
    
    if frame1 is not None and frame2 is not None:
        # Concatenate frames horizontally
        combined_frame = np.hstack((frame1, frame2))

        # Process the combined frame with your model and other operations
        results = model(combined_frame, stream=True)

        # Coordinates and bounding boxes
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # Bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Draw bounding box
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Confidence
                confidence = math.ceil((box.conf[0]*100))/100
                print("Confidence --->", confidence)

                # Class name
                cls = int(box.cls[0])
                if cls < len(classNames):
                    print("Class name -->", classNames[cls])
                    # Object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    
                    frequency, rate = get_sound_params([x1, y1, x2, y2])

                    # Play the sound (consider threading or non-blocking methods if needed)
                    for _ in range(rate):
                        winsound.Beep(frequency, 100)  # 100 ms duration for each beep

                    cv2.putText(combined_frame, classNames[cls], org, font,
                                fontScale, color, thickness)
                else:
                    print("Detected class index out of range:", cls)

        # Display the combined frame
        cv2.imshow('Dual Camera Stream', combined_frame)

    if cv2.waitKey(1) == ord('q'):
        break

cv2.destroyAllWindows()
