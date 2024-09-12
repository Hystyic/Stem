import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
from cvlib.object_detection import detect_common_objects
import time
import requests
from gtts import gTTS
import os

# Stereo Camera URLs
url1 = "https://www.ikea.com/in/en/images/products/hemlagad-pot-with-lid-black__0789061_pe763799_s5.jpg"
url2 = "https://www.ikea.com/in/en/images/products/hemlagad-pot-with-lid-black__0789061_pe763799_s5.jpg"
  # Right Camera

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
    print("Playing sound on left.")  # Debug statement
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 1.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_left > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("left.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_left = current_time

def play_sound_on_right(distance, current_time):
    global last_played_right
    print("Playing sound on right.")  # Debug statement
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 1.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_right > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("right.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_right = current_time

# Function to query Google Lens via SERPApi
def query_google_lens(image_path):
    print(f"Querying Google Lens with {image_path}")  # Debug statement
    api_key = ''  # Replace with your SERPAPI key
    endpoint = 'https://serpapi.com/search?engine=google_lens'
    params = {
        'engine': 'google_lens',
        'api_key': api_key
    }
    with open(image_path, 'rb') as image_file:
        files = {'image': image_file}
        response = requests.post(endpoint, files=files, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error in querying Google Lens. Status Code: {response.status_code}, Response: {response.text}")  # Enhanced error logging
            return None# Function to perform voiceover
        
def perform_voiceover(text):
    print(f"Performing voiceover: {text}")  # Debug statement
    tts = gTTS(text=text, lang='en')
    tts.save("description.mp3")
    os.system("mpg321 description.mp3")

# Fetch and decode image from URL
def fetch_image_from_url(url):
    print(f"Fetching image from URL: {url}")  # Debug statement
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image

# Compute depth map using stereo vision
def compute_depth_map(left_img, right_img):
    print("Computing depth map.")  # Debug statement
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

# Main function to process and display camera feeds
def run():
    print("Starting main run function.")  # Debug statement
    cv2.namedWindow("Stereo Camera Feed", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            current_time = time.time()
            left_image = fetch_image_from_url(url1)
            right_image = fetch_image_from_url(url2)
            depth_map = compute_depth_map(left_image, right_image)

            bbox_left, label_left, conf_left = detect_common_objects(left_image)
            bbox_right, label_right, conf_right = detect_common_objects(right_image)

            avg_depth_left = avg_depth_right = 0
            for (bbox, label, image) in [(bbox_left, label_left, left_image), (bbox_right, label_right, right_image)]:
                for box, lbl in zip(bbox, label):
                    x1, y1, x2, y2 = box
                    depth = depth_map[y1:y2, x1:x2].mean()
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label_with_distance = f"{lbl}: {depth:.2f} meters"
                    cv2.putText(image, label_with_distance, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if lbl == 'person':
                        print(f"Person detected: {lbl} at {depth:.2f} meters.")  # Debug statement
                        if image is left_image:
                            avg_depth_left = depth
                        else:
                            avg_depth_right = depth

            if 'person' in label_left:
                play_sound_on_left(avg_depth_left, current_time)
            if 'person' in label_right:
                play_sound_on_right(avg_depth_right, current_time)

            combined_image = np.hstack((left_image, right_image))
            cv2.imshow("Stereo Camera Feed", combined_image)

            key = cv2.waitKey(5)
            if key == ord('q'):
                print("Exiting.")  # Debug statement
                break
            elif key == ord('c'):
                cv2.imwrite('captured_image.jpg', left_image)
                response = query_google_lens('captured_image.jpg')
                if response:
                    description = response.get('description', 'No description found.')
                    perform_voiceover(description)

        except Exception as e:
            print(f"Error: {e}")  # Debug statement
            break

    cv2.destroyAllWindows()

# Running the process
if __name__ == '__main__':
    print("Starting the program.")  # Debug statement
    try:
        process = multiprocessing.Process(target=run)
        process.start()
        process.join()
    except KeyboardInterrupt:
        print("Keyboard Interrupt.")  # Debug statement
    finally:
        cv2.destroyAllWindows()
        print("Program ended.")  # Debug statement
