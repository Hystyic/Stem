import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
from cvlib.object_detection import detect_common_objects
import time

# Stereo Camera URLs
url1 = 'http://192.168.0.207/640x480.jpg'  # Left Camera
url2 = 'http://192.168.0.154/640x480.jpg'  # Right Camera

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

# Fetch and decode image from URL
def fetch_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image

# Compute depth map using stereo vision
def compute_depth_map(left_img, right_img):
    # [existing code]
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image

# Main function to process and display camera feeds
def run():
    cv2.namedWindow("Stereo Camera Feed", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            current_time = time.time()
            left_image = fetch_image_from_url(url1)
            right_image = fetch_image_from_url(url2)
            depth_map = compute_depth_map(left_image, right_image)

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

            key = cv2.waitKey(5)
            if key == ord('q'):
                break
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
