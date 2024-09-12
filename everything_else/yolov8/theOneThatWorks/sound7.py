import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
from cvlib.object_detection import detect_common_objects

# Stereo Camera URLs
url1 = 'http://192.168.0.207/640x480.jpg'  # Left Camera
url2 = 'http://192.168.0.154/640x480.jpg'  # Right Camera

# Function to play sound
def play_sound_on_left():
    wave_obj = sa.WaveObject.from_wave_file("left.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

def play_sound_on_right():
    wave_obj = sa.WaveObject.from_wave_file("right.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Constants
PERSON_WIDTH = 0.5
CALIBRATED_FOCAL_LENGTH = 500
STEREO_BASELINE = 0.3

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

    # Calculate depth map
    depth_map = np.zeros_like(disparity, np.float32)
    valid_disp = (disparity > 0) & (disparity < stereo.getNumDisparities())
    depth_map[valid_disp] = (STEREO_BASELINE * CALIBRATED_FOCAL_LENGTH) / disparity[valid_disp]

    return depth_map

# Main function to process and display camera feeds
def run():
    cv2.namedWindow("Stereo Camera Feed", cv2.WINDOW_AUTOSIZE)
    while True:
        try:
            left_image = fetch_image_from_url(url1)
            right_image = fetch_image_from_url(url2)
            depth_map = compute_depth_map(left_image, right_image)

            # Detect objects in both images
            bbox_left, label_left, conf_left = detect_common_objects(left_image)
            bbox_right, label_right, conf_right = detect_common_objects(right_image)

            # Process and annotate both images
            for (bbox, label, image) in [(bbox_left, label_left, left_image), (bbox_right, label_right, right_image)]:
                for box, lbl in zip(bbox, label):
                    x1, y1, x2, y2 = box
                    depth = depth_map[y1:y2, x1:x2].mean()
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    label_with_distance = f"{lbl}: {depth:.2f} meters"
                    cv2.putText(image, label_with_distance, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Concatenate images for stereo view
            combined_image = np.hstack((left_image, right_image))
            cv2.imshow("Stereo Camera Feed", combined_image)

            # Play sound based on detection
            if label_left and depth<5:
                play_sound_on_left()
            if label_right and depth<5:
                play_sound_on_right()

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