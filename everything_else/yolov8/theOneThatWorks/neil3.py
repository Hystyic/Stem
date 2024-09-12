import cv2
import numpy as np
import urllib.request
import time
import threading
import winsound
from ultralytics import YOLO
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

# Function to initialize and return a new YOLO model
def get_yolo_model():
    return YOLO("yolo-Weights/yolov8n.pt")

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
              "teddy bear", "hair drier", "toothbrush"]

# Frame buffers
frame_buffer1 = deque(maxlen=5)
frame_buffer2 = deque(maxlen=5)

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
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    return frame
    except ConnectionResetError:
        print(f"Connection was reset for {stream_url}. Reconnecting...")
        time.sleep(1)

# Function to play beep based on bounding box size
def play_beep(box):
    area = (box[2] - box[0]) * (box[3] - box[1])
    frequency = 1500
    duration = 100
    rate = 2

    if area > 50000:
        frequency = 500
        rate = 10
    elif area > 20000:
        frequency = 1000
        rate = 5

    for _ in range(rate):
        winsound.Beep(frequency, duration)
        time.sleep(1 / rate)

# Function to process a frame for object detection
def process_frame(frame, classNames):
    local_model = get_yolo_model()
    results = local_model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cls = int(box.cls[0])
            cv2.putText(frame, f"{classNames[cls]}: {math.ceil(box.conf[0]*100)/100}", 
                        (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            play_beep([x1, y1, x2, y2])
    return frame

# Thread function to continuously fetch frames from a stream
def fetch_stream_frames(stream_url, frame_buffer):
    while True:
        frame = fetch_frame(stream_url)
        if frame is not None:
            frame_buffer.append(frame)
        time.sleep(0.01)

# Main execution function
def run():
    threading.Thread(target=fetch_stream_frames, args=(stream_url1, frame_buffer1), daemon=True).start()
    threading.Thread(target=fetch_stream_frames, args=(stream_url2, frame_buffer2), daemon=True).start()

    with ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            if frame_buffer1 and frame_buffer2:
                future1 = executor.submit(process_frame, frame_buffer1[-1], classNames)
                future2 = executor.submit(process_frame, frame_buffer2[-1], classNames)

                processed_frame1 = future1.result()
                processed_frame2 = future2.result()

                combined_frame = np.hstack((processed_frame1, processed_frame2))
                cv2.imshow('Stereo Object Detection', combined_frame)

            if cv2.waitKey(1) == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()