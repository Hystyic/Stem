import cv2
import math
import urllib.request
from ultralytics import YOLO
import numpy as np
import time
import threading
import winsound

# Initialize YOLOv8 model
model = YOLO("yolo-Weights/yolov8n.pt")

# URLs of the MJPEG streams
stream_url1 = 'http://192.168.137.141/640x480.mjpeg'
stream_url2 = 'http://192.168.137.175/640x480.mjpeg'

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
        print(f"Connection was reset for {stream_url}. Reconnecting...")
        time.sleep(1)

# Function to play beep based on bounding box size
def play_beep(box):
    # Calculate the area of the bounding box
    area = (box[2] - box[0]) * (box[3] - box[1])

    # Map the area to a frequency and rate
    frequency = 1500  # default frequency
    duration = 100    # default duration of each beep in milliseconds
    rate = 2          # default beeps per second

    if area > 50000:     # Very close object
        frequency = 500  # Lower frequency
        rate = 10        # Faster beeping
    elif area > 20000:   # Moderately close object
        frequency = 1000
        rate = 5

    # Play the beep sound
    for _ in range(rate):
        winsound.Beep(frequency, duration)
        time.sleep(1/rate)

# Function to process and display frames from a stream
def process_stream(stream_url, model, classNames):
    while True:
        frame = fetch_frame(stream_url)
        if frame is not None:
            results = model(frame, stream=True)

            for r in results:
                boxes = r.boxes

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # Get class name
                    cls = int(box.cls[0])

                    # Object details
                    org = [x1, y1]
                    cv2.putText(frame, f"{classNames[cls]}: {math.ceil(box.conf[0]*100)/100}", org, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                    # Play beep sound based on the size of the bounding box
                    play_beep([x1, y1, x2, y2])

            cv2.imshow(f'MJPEG Stream Object Detection {stream_url}', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

# Main execution
if __name__ == "__main__":
    # Create threads for each stream
    thread1 = threading.Thread(target=process_stream, args=(stream_url1, model, classNames))
    thread2 = threading.Thread(target=process_stream, args=(stream_url2, model, classNames))

    # Start threads
    thread1.start()
    thread2.start()

    # Wait for threads to finish
    thread1.join()
    thread2.join()  