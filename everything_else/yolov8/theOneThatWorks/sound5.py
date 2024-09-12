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
    depth_map[disparity > 0] = (STEREO_BASELINE * CALIBRATED_FOCAL_LENGTH) / disparity[disparity > 0]

    return depth_map

# Main function to process each camera feed
def run(camera_url, window_name, sound_function, is_left_camera):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        left_image = fetch_image_from_url(url1)
        right_image = fetch_image_from_url(url2)
        depth_map = compute_depth_map(left_image, right_image)

        # Process the appropriate image based on the camera
        image_to_process = left_image if is_left_camera else right_image

        # Detect objects
        bbox, label, conf = detect_common_objects(image_to_process)

        # Draw bounding boxes and labels with distances
        for box, lbl in zip(bbox, label):
            x1, y1, x2, y2 = box
            object_width_in_pixels = x2 - x1
            distance = (PERSON_WIDTH * CALIBRATED_FOCAL_LENGTH) / object_width_in_pixels
            cv2.rectangle(image_to_process, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label_with_distance = f"{lbl}: {distance:.2f} meters"
            cv2.putText(image_to_process, label_with_distance, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow(window_name + " - Detection", image_to_process)

        # Print debug information
        print("Detected labels:", label)

        # Play sound based on camera detection
        if label and 'person' in label:
            print("Playing sound...")
            sound_function()

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

# Running the processes
if __name__ == '__main__':
    print("started")
    try:
        process1 = multiprocessing.Process(target=run, args=(url1, "Camera 1", play_sound_on_left, True))
        process2 = multiprocessing.Process(target=run, args=(url2, "Camera 2", play_sound_on_right, False))

        process1.start()
        process2.start()

        process1.join()
        process2.join()
    except KeyboardInterrupt:
        pass