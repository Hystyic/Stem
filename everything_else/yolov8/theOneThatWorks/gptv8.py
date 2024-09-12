import cv2
import urllib.request
import numpy as np
import torch
import multiprocessing
import simpleaudio as sa

# URLs for the camera feeds
url1 = 'http://192.168.0.142/640x480.jpg'
url2 = 'http://192.168.0.154/640x480.jpg'

# Load YOLOv8 model
model = torch.hub.load('./ultralytics', 'yolov8')  # Replace 'yolov8' with the specific model you want to use

def play_sound_on_left():
    wave_obj = sa.WaveObject.from_wave_file("left.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

def play_sound_on_right():
    wave_obj = sa.WaveObject.from_wave_file("right.wav")
    play_obj = wave_obj.play()
    play_obj.wait_done()

def run(camera_url, window_name, sound_function):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(camera_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Object detection with YOLOv8
        results = model(im)
        im = np.squeeze(results.render())  # Render detections on image

        # Display image with detections
        cv2.imshow(window_name + " - Detection", im)

        # Check for 'person' detection
        labels = [detection[0] for detection in results.xyxy[0] if int(detection[5]) == 0]
        if 'person' in labels:
            print("Playing sound...")
            sound_function()

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("started")
    try:
        process1 = multiprocessing.Process(target=run, args=(url1, "Camera 1", play_sound_on_left))
        process2 = multiprocessing.Process(target=run, args=(url2, "Camera 2", play_sound_on_right))

        process1.start()
        process2.start()

        process1.join()
        process2.join()
    except KeyboardInterrupt:
        pass
