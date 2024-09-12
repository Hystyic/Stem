import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
from cvlib.object_detection import detect_common_objects
import time
import requests
from gtts import gTTS
import io
from google.cloud import vision
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./stem-405404-3ddc4f499b6f.json"

# Stereo Camera URLs
url1 = 'http://192.168.180.152/640x480.jpg'
url2 = 'http://192.168.180.244/640x480.jpg'
# Constants

# Constants
PERSON_WIDTH = 0.5
CALIBRATED_FOCAL_LENGTH = 500
STEREO_BASELINE = 0.3
MAX_PLAYBACK_INTERVAL = 1.0  # seconds
MIN_PLAYBACK_INTERVAL = 0.1  # seconds

# Last played time for sounds
last_played_left = 0
last_played_right = 0

# Function to play sound with conditional timing
def play_sound_on_left(distance, current_time):
    global last_played_left
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 1.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_left > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("left.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_left = current_time

def play_sound_on_right(distance, current_time):
    global last_played_right
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 1.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_right > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("right.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_right = current_time

# Function to perform voiceover
def perform_voiceover(text):
    tts = gTTS(text=text, lang='en')
    tts.save("description.mp3")
    os.system("mpg321 description.mp3")

# Fetch and decode image from URL
def fetch_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image

# Compute depth map using stereo vision
def compute_depth_map(left_img, right_img):
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=16*5,
        blockSize=5,
        P1=8 * 3 * 5 ** 2,
        P2=32 * 3 * 5 ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    disparity = stereo.compute(left_gray, right_gray)

    depth_map = np.zeros_like(disparity, np.float32)
    valid_disp = (disparity > 0) & (disparity < stereo.getNumDisparities())
    depth_map[valid_disp] = (STEREO_BASELINE * CALIBRATED_FOCAL_LENGTH) / disparity[valid_disp]

    return depth_map

# Function to detect objects using Google Cloud Vision API
def detect_objects_google_vision(image_path):
    client = vision.ImageAnnotatorClient()

    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)
    response = client.object_localization(image=image)
    objects = response.localized_object_annotations

    return objects

# Main function to process and display camera feeds
def run():
    cv2.namedWindow("Stereo Camera Feed", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            current_time = time.time()
            left_image = fetch_image_from_url(url1)
            right_image = fetch_image_from_url(url2)
            depth_map = compute_depth_map(left_image, right_image)

            bbox_left, label_left, conf_left = detect_common_objects(left_image)
            bbox_right, label_right, conf_right = detect_common_objects(right_image)

            if not bbox_left:  # Fallback to Google Cloud Vision API
                objects = detect_objects_google_vision('captured_image.jpg')
                for object_ in objects:
                    print(f"{object_.name} (confidence: {object_.score})")
                    # Add any specific processing you need here
            else:
                # Process using data from OpenCV's detect_common_objects
                # Your existing processing code here

                combined_image = np.hstack((left_image, right_image))
                cv2.imshow("Stereo Camera Feed", combined_image)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            elif key == ord('c'):
                cv2.imwrite('captured_image.jpg', left_image)
                # More processing if needed

        except Exception as e:
            print(f"Error: {e}")
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    multiprocessing.Process(target=run).start()
