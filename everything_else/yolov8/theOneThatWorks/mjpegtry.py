import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
import time
import io
import os
import time
from google.cloud import vision
from gtts import gTTS
from cvlib.object_detection import detect_common_objects
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./stem-405404-3ddc4f499b6f.json"

# Stereo Camera URLs


if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("No CUDA device detected. Falling back to CPU.")
else:
    cv2.cuda.setDevice(0)  # Set the CUDA device. Assuming you have one CUDA-capable GPU.
    print("cuda being used lesgo")
url1 = 'http://192.168.0.207/640x480.mjpeg'  # Camera 1
# url2 = 'http://192.168.0.207/640x480.mjpeg'  # Camera 2

MAX_PLAYBACK_INTERVAL = 0.1  # seconds
MIN_PLAYBACK_INTERVAL = 0.1  # seconds

# Last played time for sounds
last_played_left = 0
last_played_right = 0

# Function to play sound with conditional timing
def play_sound_on_left(distance, current_time):
    global last_played_left
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 2.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_left > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("left.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_left = current_time

def play_sound_on_right(distance, current_time):
    global last_played_right
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 2.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_right > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("right.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_right = current_time

# Fetch and decode image from URL
def fetch_image_from_url(url):
    stream = urllib.request.urlopen(url)
    bytes = b''
    while True:
        bytes += stream.read(1024)
        a = bytes.find(b'\xff\xd8')  # JPEG start
        b = bytes.find(b'\xff\xd9')  # JPEG end
        if a != -1 and b != -1:
            jpg = bytes[a:b+2]  # Extract the JPEG frame
            bytes = bytes[b+2:]  # Remove the processed frame from the buffer
            image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
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

def run():
    cv2.namedWindow("Stereo Camera Feed", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            left_image = fetch_image_from_url(url1)
            # depth_map, disp_norm = compute_depth_map(left_image, right_image)

            # Concatenate images for stereo view
            cv2.imshow("Stereo Camera Feed", left_image)
            # cv2.imshow("Disparity Normalized", disp_norm)  # Display normalized disparity map

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
        except Exception as e:
            time.sleep(1)
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
