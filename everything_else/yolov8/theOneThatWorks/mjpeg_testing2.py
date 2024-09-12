import cv2
import numpy as np
import urllib.request
import multiprocessing
import simpleaudio as sa
import time
import os
from google.cloud import vision
from gtts import gTTS
import winsound

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./stem-405404-3ddc4f499b6f.json"

# Stereo Camera URLs
url1 = 'http://192.168.0.207/640x480.mjpeg'  # Camera 1
url2 = 'http://192.168.0.207/640x480.mjpeg'  # Camera 2

# Constants
CALIBRATED_FOCAL_LENGTH = 500  # Adjust as per calibration
STEREO_BASELINE = 0.3  # Adjust as per physical measurement (in meters)

# Last played time for sounds
last_played_left = 0
last_played_right = 0

# New global variables for playback rates (in beats per second)
left_sound_rate = 0
right_sound_rate = 0

# Initialize CUDA device
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)
    print("CUDA device detected and set.")
else:
    print("No CUDA device detected. Falling back to CPU.")

# Function to play sound on the left at a given rate
import winsound
import time

def play_left_sound(rate):
    if rate > 0:
        duration_per_beep = int(1000 / rate)  # Duration of each beep in milliseconds
        for _ in range(rate):
            winsound.Beep(1000, duration_per_beep)  
            time.sleep(duration_per_beep / 1000.0)  # Pause between beeps


# Function to play sound on the right at a given rate


# def play_right_sound(rate):
#     global last_played_right
#     if rate > 0:
#         current_time = time.time()
#         interval = 1.0 / rate
#         if current_time - last_played_right >= interval:
#             wave_obj = sa.WaveObject.from_wave_file("right.wav")
#             play_obj = wave_obj.play()
#             play_obj.wait_done()
#             last_played_right = current_time

# Function to check if URL stream is available
def is_url_accessible(url):
    try:
        urllib.request.urlopen(url, timeout=5)  # timeout in 5 seconds
        return True
    except urllib.error.URLError:
        return False

# Fetch and decode image from URL, then upload to GPU
def fetch_image_from_url(url):
    try:
        with urllib.request.urlopen(url) as stream:
            bytes = b''
            while True:
                bytes += stream.read(1024)
                a = bytes.find(b'\xff\xd8')  # Start of JPEG
                b = bytes.find(b'\xff\xd9')  # End of JPEG
                if a != -1 and b != -1:
                    jpg = bytes[a:b+2]
                    bytes = bytes[b+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame
    except ConnectionResetError:
        print(f"Connection was reset for {url}. Reconnecting...")
        time.sleep(1)


# Compute depth map using stereo vision (GPU optimized)
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
        object_desc = objects[0].name
        print("Detected object:", object_desc)
        perform_voiceover(f"Detected object is {object_desc}.")
    else:
        print("No objects detected.")

# Main function to process and display camera feeds
# Main function to process and display camera feeds
def run():
    # ...
    global left_sound_rate, right_sound_rate

    # Initialize sound rates to a default value
    left_sound_rate = 0  # Default rate, can be adjusted as needed
    right_sound_rate = 0  # Default rate, can be adjusted as needed

    while True:
        # ...
        left_image = fetch_image_from_url(url1)
        right_image = fetch_image_from_url(url2)

        if left_image is not None and right_image is not None:
            depth_map, disp_norm = compute_depth_map(left_image, right_image)
            # ...

            objects_left = detect_objects_google_vision(left_image)
            objects_right = detect_objects_google_vision(right_image)

            for objects, image in [(objects_left, left_image), (objects_right, right_image)]:
                for obj in objects:
                    vertices = [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
                    if vertices:
                        x1, y1 = int(vertices[0][0] * image.shape[1]), int(vertices[0][1] * image.shape[0])
                        x2, y2 = int(vertices[2][0] * image.shape[1]), int(vertices[2][1] * image.shape[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = obj.name
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            combined_image = np.hstack((left_image, right_image))
            cv2.imshow("Stereo Camera Feed", combined_image)
            cv2.imshow("Disparity Normalized", disp_norm)

            # Rest of your code...


            key = cv2.waitKey(5)
            if key == ord('q'):
                break
            elif key == ord('u'):
                left_sound_rate = 20  # 20 bps for left sound
            elif key == ord('h'):
                left_sound_rate = 5   # 5 bps for left sound
            elif key == ord('b'):
                left_sound_rate = 1   # 1 bps for left sound
            elif key == ord('i'):
                right_sound_rate = 20 # 20 bps for right sound
            elif key == ord('j'):
                right_sound_rate = 5  # 5 bps for right sound
            elif key == ord('n'):
                right_sound_rate = 1  # 1 bps for right sound

            # Play sounds based on the current rates
            play_left_sound(left_sound_rate)
            # play_right_sound(right_sound_rate)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("Stereo camera processing started.")
    try:
        process = multiprocessing.Process(target=run)
        process.start()
        process.join()
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
