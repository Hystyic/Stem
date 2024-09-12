import cv2
import numpy as np
import urllib.request
import multiprocessing
import simpleaudio as sa
import time
import os
from google.cloud import vision
from gtts import gTTS

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./stem-405404-3ddc4f499b6f.json"

# Stereo Camera URLs
url1 = 'http://192.168.0.154/stream'  # Camera 1
url2 = 'http://192.168.0.154/stream'  # Camera 2

# Constants
PERSON_WIDTH = 0.5
CALIBRATED_FOCAL_LENGTH = 500  # Adjust as per calibration
STEREO_BASELINE = 0.3  # Adjust as per physical measurement (in meters)
MAX_PLAYBACK_INTERVAL = 0.1  # seconds
MIN_PLAYBACK_INTERVAL = 0.1  # seconds

# Last played time for sounds
last_played_left = 0
last_played_right = 0

# Initialize CUDA device
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    cv2.cuda.setDevice(0)
    print("CUDA device detected and set.")
else:
    print("No CUDA device detected. Falling back to CPU.")

# Function to play sound on the left
def play_sound_on_left(distance, current_time):
    global last_played_left
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 2.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_left > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("left.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_left = current_time

# Function to play sound on the right
def play_sound_on_right(distance, current_time):
    global last_played_right
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 2.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    if current_time - last_played_right > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file("right.wav")
        play_obj = wave_obj.play()
        play_obj.wait_done()
        last_played_right = current_time

# Fetch and decode image from URL, then upload to GPU
def fetch_image_from_url_gpu(url):
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    image_gpu = cv2.cuda_GpuMat()
    image_gpu.upload(image)
    return image_gpu

# Compute depth map using stereo vision (GPU optimized)
def compute_depth_map_gpu(left_img_gpu, right_img_gpu):
    # Convert to grayscale on GPU
    left_gray_gpu = cv2.cuda.cvtColor(left_img_gpu, cv2.COLOR_BGR2GRAY)
    right_gray_gpu = cv2.cuda.cvtColor(right_img_gpu, cv2.COLOR_BGR2GRAY)

    # Create StereoSGBM object
    stereo = cv2.cuda_StereoSGBM.create(
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

    # Compute disparity map on GPU
    disparity_gpu = stereo.compute(left_gray_gpu, right_gray_gpu)

    # Normalize disparity for visualization
    disp_norm_gpu = cv2.cuda.normalize(disparity_gpu, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Download results from GPU to CPU for further processing
    disparity = disparity_gpu.download()
    disp_norm = disp_norm_gpu.download()

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
            left_image_gpu = fetch_image_from_url_gpu(url1)
            right_image_gpu = fetch_image_from_url_gpu(url2)
            depth_map, disp_norm = compute_depth_map_gpu(left_image_gpu, right_image_gpu)

            # Download images from GPU for CPU-based processing
            left_image = left_image_gpu.download()
            right_image = right_image_gpu.download()

            # Detect objects in both images (CPU-based)
            objects_left = detect_objects_google_vision(left_image)
            objects_right = detect_objects_google_vision(right_image)

            # Process and annotate both images
            for objects, image in [(objects_left, left_image), (objects_right, right_image)]:
                for obj in objects:
                    vertices = [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
                    if vertices:
                        x1, y1 = int(vertices[0][0] * image.shape[1]), int(vertices[0][1] * image.shape[0])
                        x2, y2 = int(vertices[2][0] * image.shape[1]), int(vertices[2][1] * image.shape[0])
                        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                        label = obj.name
                        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

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
