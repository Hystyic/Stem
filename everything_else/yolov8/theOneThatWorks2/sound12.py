import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
from ultralytics import YOLO
import torch
from google.cloud import vision
from gtts import gTTS
import time
import os


STEREO_BASELINE = 0.1  
CALIBRATED_FOCAL_LENGTH = 800  
MAX_PLAYBACK_INTERVAL = 5  
MIN_PLAYBACK_INTERVAL = 1  
last_played_left = 0  
last_played_right = 0  


model = YOLO("yolov8n.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


url1 = 'http://192.168.0.142/640x480.jpg'  
url2 = 'http://192.168.0.142/640x480.jpg'  


def is_stream_available(stream_url):
    try:
        response = urllib.request.urlopen(stream_url)
        return response.status == 200
    except urllib.request.URLError:
        return False


def play_sound(distance, current_time, camera_side):
    global last_played_left, last_played_right
    playback_interval = MAX_PLAYBACK_INTERVAL - min(distance, 2.0) * (MAX_PLAYBACK_INTERVAL - MIN_PLAYBACK_INTERVAL)
    last_played = last_played_left if camera_side == 'left' else last_played_right
    sound_file = "left.wav" if camera_side == 'left' else "right.wav"

    if current_time - last_played > playback_interval:
        wave_obj = sa.WaveObject.from_wave_file(sound_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
        if camera_side == 'left':
            last_played_left = current_time
        else:
            last_played_right = current_time


def fetch_image_from_url(url):
    resp = urllib.request.urlopen(url)
    image = np.array(bytearray(resp.read()), dtype=np.uint8)
    image = cv2.imdecode(image, -1)
    return image


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
    disp_norm = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_map = np.zeros_like(disparity, np.float32)
    valid_disp = (disparity > 0) & (disparity < stereo.getNumDisparities())
    depth_map[valid_disp] = (STEREO_BASELINE * CALIBRATED_FOCAL_LENGTH) / (disparity[valid_disp] + 1e-6)
    return depth_map, disp_norm


def detect_objects_yolo(image, model):
    class_ids, scores, boxes = model.detect(image, confThreshold=0.4, nmsThreshold=0.3)
    return class_ids, scores, boxes


def perform_voiceover(text):
    tts = gTTS(text=text, lang='en')
    tts.save("description.mp3")
    os.system("mpg321 description.mp3")


def run(camera_url_left, camera_url_right, window_name):
    cap_left = cv2.VideoCapture(camera_url_left)
    cap_right = cv2.VideoCapture(camera_url_right)

    while cap_left.isOpened() and cap_right.isOpened():
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        current_time = time.time()

        if ret_left and ret_right:
            
            frame_left_processed = cv2.resize(frame_left, (640, 640))
            frame_right_processed = cv2.resize(frame_right, (640, 640))

            
            class_ids_left, scores_left, boxes_left = detect_objects_yolo(frame_left_processed, model)
            class_ids_right, scores_right, boxes_right = detect_objects_yolo(frame_right_processed, model)

            
            depth_map, disp_norm = compute_depth_map(frame_left_processed, frame_right_processed)

            
            for class_id, score, box in zip(class_ids_left, scores_left, boxes_left):
                if class_id == 0:  
                    x, y, w, h = box
                    depth = depth_map[y:y+h, x:x+w].mean()
                    play_sound(depth, current_time, 'left')

            
            combined_image = np.hstack((frame_left_processed, frame_right_processed))
            cv2.imshow(window_name, combined_image)
            cv2.imshow("Disparity Normalized", disp_norm)

            key = cv2.waitKey(5)
            if key == ord('q'):
                break

        else:
            print("Failed to read from cameras")
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if is_stream_available(url1) and is_stream_available(url2):
        process1 = multiprocessing.Process(target=run, args=(url1, url2, "Stereo Camera Feed"))
        process1.start()
        process1.join()
    else:
        print("One or both of the video streams are not available.")