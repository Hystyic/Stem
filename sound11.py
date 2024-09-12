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

# Stereo Camera URLs


if cv2.cuda.getCudaEnabledDeviceCount() == 0:
    print("No CUDA device detected. Falling back to CPU.")
else:
    cv2.cuda.setDevice(0)  # Set the CUDA device. Assuming you have one CUDA-capable GPU.
    print("cuda being used lesgo")
url1 = 'http://192.168.0.207/640x480.mjpeg'  # Camera 1
url2 = 'http://192.168.0.207/640x480.mjpeg'  # Camera 2

# Constants
PERSON_WIDTH = 0.5
CALIBRATED_FOCAL_LENGTH = 500  # Adjust as per calibration
STEREO_BASELINE = 0.3  # Adjust as per physical measurement (in meters)
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
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image

# Compute depth map using stereo vision
def compute_depth_map(left_img, right_img):
    # Convert to grayscale
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM object
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

    # Compute disparity map
    disparity = stereo.compute(left_gray, right_gray)

    # Normalize disparity for visualization
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Calculate depth map
    depth_map = np.zeros_like(disparity, np.float32)
    valid_disp = (disparity > 0) & (disparity < stereo.getNumDisparities())
    depth_map[valid_disp] = (STEREO_BASELINE * CALIBRATED_FOCAL_LENGTH) / (disparity[valid_disp] + 1e-6)

    return depth_map, disp_norm

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
        # Assuming you want to use the first detected object for simplicity
        object_desc = objects[0].name
        print("Detected object:", object_desc)
        # Generate and perform voiceover
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
            depth_map, disp_norm = compute_depth_map(left_image, right_image)

            # Detect objects in both images
            bbox_left, label_left, conf_left = detect_common_objects(left_image)
            bbox_right, label_right, conf_right = detect_common_objects(right_image)

            # Process and annotate both images
            avg_depth_left = avg_depth_right = 0
            for (bbox, label, image) in [(bbox_left, label_left, left_image), (bbox_right, label_right, right_image)]:
                for box, lbl in zip(bbox, label):
                    x1, y1, x2, y2 = box
                    depth = depth_map[y1:y2, x1:x2].mean()
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label_with_distance = f"{lbl}: {depth:.2f} meters"
                    cv2.putText(image, label_with_distance, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    if lbl == 'person':
                        if image is left_image:
                            avg_depth_left = depth
                        else:
                            avg_depth_right = depth

            # Play sound based on detection
            if 'person' in label_left:
                play_sound_on_left(avg_depth_left, current_time)
            if 'person' in label_right:
                play_sound_on_right(avg_depth_right, current_time)

            # Concatenate images for stereo view
            combined_image = np.hstack((left_image, right_image))
            cv2.imshow("Stereo Camera Feed", combined_image)
            cv2.imshow("Disparity Normalized", disp_norm)  # Display normalized disparity map

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Capture current frame from left camera and analyze
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
