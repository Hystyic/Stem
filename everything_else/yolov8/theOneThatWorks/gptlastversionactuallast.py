import cv2
import numpy as np
import threading
import winsound
from ultralytics import YOLO
import urllib.request
import math

# Initialize YOLOv8 model
model = YOLO("yolo-Weights/yolov8x.pt")

# URLs of the MJPEG streams
stream_urls = ['http://192.168.137.102/640x480.mjpeg', 'http://192.168.137.108/640x480.mjpeg']

# Object classes (list of class names)
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

# Function to fetch a single frame from a camera
def fetch_frame(url, frames, index):
    try:
        with urllib.request.urlopen(url, timeout=5) as stream:
            bytes = b''
            while True:
                bytes += stream.read(1024)
                a = bytes.find(b'\xff\xd8')  # Start of JPEG
                b = bytes.find(b'\xff\xd9')  # End of JPEG
                if a != -1 and b != -1:
                    jpg = bytes[a:b+2]
                    bytes = bytes[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    frames[index] = frame
                    break
    except Exception as e:
        print(f"Error fetching frame: {e}")

# Function to fetch frames in parallel from all cameras
def fetch_frames_parallel(stream_urls):
    threads = []
    frames = [None] * len(stream_urls)
    for i, url in enumerate(stream_urls):
        thread = threading.Thread(target=fetch_frame, args=(url, frames, i))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    return frames

# Function to calculate sound parameters based on detected object size and camera index
def get_sound_params(box, camera_index):
    area = (box[2] - box[0]) * (box[3] - box[1])
    if camera_index == 0:  # First camera
        if area > 50000:
            return 500, 10
        elif area > 20000:
            return 1000, 5
        else:
            return 1500, 2
    else:  # Second camera
        if area > 50000:
            return 550, 10
        elif area > 20000:
            return 1050, 5
        else:
            return 1550, 2

# Function to play sound
def play_sound(frequency, rate):
    for _ in range(rate):
        winsound.Beep(frequency, 100)

# Main loop to process frames and detect objects
while True:
    frames = fetch_frames_parallel(stream_urls)
    if all(frame is not None and frame.any() for frame in frames):
        combined_frame = np.hstack(frames)
        results = model(combined_frame, stream=True)

        frame_width = frames[0].shape[1]

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                camera_index = 0 if x1 < frame_width else 1  # Determine the camera index
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                confidence = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                if 0 <= cls < len(classNames):
                    cv2.putText(combined_frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    frequency, rate = get_sound_params([x1, y1, x2, y2], camera_index)
                    threading.Thread(target=play_sound, args=(frequency, rate)).start()

        cv2.imshow('Dual Camera Stream', combined_frame)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break