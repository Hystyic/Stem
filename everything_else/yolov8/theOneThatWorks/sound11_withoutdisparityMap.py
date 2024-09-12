import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
import time
import io
import os
from google.cloud import vision
from gtts import gTTS
from cvlib.object_detection import detect_common_objects
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./stem-405404-3ddc4f499b6f.json"

# Check for CUDA device
if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("No CUDA device detected. Falling back to CPU.")
else:
    cv2.cuda.setDevice(0)  # Set the CUDA device. Assuming you have one CUDA-capable GPU.
    print("CUDA being used lesgo")

# Stereo Camera URLs
url1 = 'http://192.168.0.142/640x480.jpg'  # Camera 1
url2 = 'http://192.168.0.142/640x480.jpg'  # Camera 2

# Constants for sound playback
MAX_PLAYBACK_INTERVAL = 0.1  # seconds
MIN_PLAYBACK_INTERVAL = 0.1  # seconds

# Last played time for sounds
last_played_left = 0
last_played_right = 0

# Function to play sound with conditional timing
def play_sound_on_left(current_time):
    global last_played_left
    if current_time - last_played_left > MAX_PLAYBACK_INTERVAL:
        wave_obj = sa.WaveObject.from_wave_file("left.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_left = current_time

def play_sound_on_right(current_time):
    global last_played_right
    if current_time - last_played_right > MAX_PLAYBACK_INTERVAL:
        wave_obj = sa.WaveObject.from_wave_file("right.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_right = current_time

# Fetch and decode image from URL
def fetch_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image

# Function to detect objects using Google Cloud Vision API
def detect_objects_google_vision(image):
    client = vision.ImageAnnotatorClient()
    content = cv2.imencode('.jpg', image)[1].tobytes()
    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations
    return objects

# Function for text-to-speech voiceover
def perform_voiceover(text):
    tts = gTTS(text=text, lang='en')
    tts.save("description.mp3")
    os.system("mpg321 description.mp3")

# Function to process and analyze captured frame
def process_captured_frame(frame):
    objects = detect_objects_google_vision(frame)
    if objects:
        object_desc = objects[0].name
        print("Detected object:", object_desc)
        perform_voiceover(f"Detected object is {object_desc}.")
    else:
        print("No objects detected.")

# Main function to process and display camera feeds
def run():
    cv2.namedWindow("Stereo Camera Feed", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            current_time = time.time()
            left_image = fetch_image_from_url(url1)
            right_image = fetch_image_from_url(url2)

            # Detect objects in both images
            bbox_left, label_left, conf_left = detect_common_objects(left_image)
            bbox_right, label_right, conf_right = detect_common_objects(right_image)

            # Process and annotate both images
            for (bbox, label, image) in [(bbox_left, label_left, left_image), (bbox_right, label_right, right_image)]:
                for box, lbl in zip(bbox, label):
                    x1, y1, x2, y2 = box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, lbl, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Play sound based on detection
            if 'person' in label_left:
                play_sound_on_left(current_time)
            if 'person' in label_right:
                play_sound_on_right(current_time)

            # Concatenate images for stereo view
            combined_image = np.hstack((left_image, right_image))
            cv2.imshow("Stereo Camera Feed", combined_image)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            elif key == ord('c'):
                process_captured_frame(left_image)

        except Exception as e:
            print(f"Error: {e}")
            break

    cv2.destroyAllWindows()

# Running the process
if __name__ == '__main__':
    print("started")
    try:
        process = multiprocessing.Process(target=run)
        process.start()
        process.join()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
