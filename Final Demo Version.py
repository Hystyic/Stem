import cv2
import numpy as np
import threading
import winsound
from ultralytics import YOLO
import urllib.request
import math
import time
import pygame
import numpy as np

# Initialize YOLOv8 model
model = YOLO("yolo-Weights/yolov8x.pt")
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)

# URLs of the MJPEG streams
stream_urls = ['http://192.168.137.43/640x480.mjpeg', 'http://192.168.137.43/640x480.mjpeg']

# Initialize Pygame for sound
pygame.init()
pygame.mixer.init()

# Define sound frequencies and rates for each key
sound_params = {
    't': {'frequency': 500, 'rate': 1, 'channel': 'left'},
    'u': {'frequency': 500, 'rate': 1, 'channel': 'right'},
    'g': {'frequency': 1000, 'rate': 2, 'channel': 'left'},
    'h': {'frequency': 1000, 'rate': 2, 'channel': 'right'},
    'b': {'frequency': 2000, 'rate': 3, 'channel': 'both'}
}

def play_stereo_sound(frequency, rate, channel):
    sample_rate = 44100
    duration = 0.1  # Duration of each beep

    # Generate a sound wave
    t = np.linspace(0, duration, int(sample_rate * duration))
    wave = np.sin(frequency * 2 * np.pi * t)

    # Set sound to the specified channel
    if channel == 'left':
        sound_wave = np.vstack((wave, np.zeros_like(wave))).T
    elif channel == 'right':
        sound_wave = np.vstack((np.zeros_like(wave), wave)).T
    else:  # both channels
        sound_wave = np.vstack((wave, wave)).T

    # Convert to 16-bit
    sound_wave = np.int16(sound_wave * 32767)

    # Create and play the sound
    sound = pygame.sndarray.make_sound(sound_wave)
    for _ in range(rate):
        sound.play()
        pygame.time.delay(int(duration * 1000))

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

def fetch_and_process_frame(url, camera_index, model, frames):
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
                    process_frame(frame, camera_index, model)
                    frames[camera_index] = frame
                    break
    except Exception as e:
        print(f"Error fetching frame: {e}")

def process_frame(frame, camera_index, model):
    results = model(frame, stream=True)
    largest_box = None
    largest_area = 0
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_box = box

    if largest_box:
        x1, y1, x2, y2 = map(int, largest_box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cls = int(largest_box.cls[0])
        if 0 <= cls < len(classNames):
            cv2.putText(frame, classNames[cls], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            frequency, rate = get_sound_params([x1, y1, x2, y2], camera_index)
            threading.Thread(target=play_sound, args=(frequency, rate)).start()

def get_sound_params(box, camera_index):
    width = box[2] - box[0]  # Width of the object
    base_freq = 1000 if camera_index == 0 else 550
    max_freq = 3000 if camera_index == 0 else 1550

    # Smoother frequency calculation
    frequency = base_freq + int((width / 200) * base_freq)
    frequency = min(max(frequency, base_freq), max_freq)

    # Dynamic rate based on object width
    if width > 200:
        rate = 1
    elif width > 100:
        rate = 2
    else:
        rate = 3
    print("width:" , width)
    return frequency, rate

def play_sound(frequency, rate):
    duration = 100  # Duration of each beep
    for _ in range(rate):
        winsound.Beep(frequency, duration)
        time.sleep(0.2)  # Increased pause between beeps


frames = [None] * len(stream_urls)
while True:
    threads = []
    for i, url in enumerate(stream_urls):
        thread = threading.Thread(target=fetch_and_process_frame, args=(url, i, model, frames))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

    if frames[0] is not None:
        cv2.imshow('Camera Stream', frames[0])

    if frames[0] is not None and frames[1] is not None:
        try:
            gray1 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[1], cv2.COLOR_BGR2GRAY)
            
            disparity = stereo.compute(gray1, gray2)
            disp_display = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            cv2.imshow('Disparity Map', disp_display)
        except Exception as e:
            print(f"Error in disparity map computation: {e}")

    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        pygame.quit()
        break
