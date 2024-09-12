import cv2
import numpy as np
import threading
import pygame
from ultralytics import YOLO
import urllib.request

# Initialize YOLOv8 model
model = YOLO("yolo-Weights/yolov8n.pt")  # Use GPU if available

# URLs of the MJPEG streams
left_stream_url = 'http://192.168.137.141/640x480.mjpeg'
right_stream_url = 'http://192.168.137.175/640x480.mjpeg'

# Initialize Pygame Mixer
pygame.mixer.init()

# Load a sound file (you need a .wav file for this)
sound = pygame.mixer.Sound("your_sound_file.wav")

def play_sound_on_channel(channel):
    if channel == "left":
        sound.set_volume(1.0, 0.0)  # Full volume on left, none on right
    elif channel == "right":
        sound.set_volume(0.0, 1.0)  # Full volume on right, none on left
    sound.play()

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

while True:
    frames = fetch_frames_parallel([left_stream_url, right_stream_url])
    if all(frame is not None and frame.any() for frame in frames):
        combined_frame = np.hstack(frames)
        results = model(combined_frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(combined_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cls = int(box.cls[0])
                if 0 <= cls < len(classNames):
                    cv2.putText(combined_frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    # Determine which camera (left or right) the object is in
                    camera_side = "left" if x1 < combined_frame.shape[1] // 2 else "right"
                    threading.Thread(target=play_sound_on_channel, args=(camera_side,)).start()

        cv2.imshow('Dual Camera Stream', combined_frame)

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break
