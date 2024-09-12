import cv2
import urllib.request
import numpy as np
from cvlib.object_detection import detect_common_objects, draw_bbox
import multiprocessing
import simpleaudio as sa

url1 = 'http://192.168.0.207/640x480.jpg' #stuck
url2 = 'http://192.168.0.154/640x480.jpg' #separate

def play_sound_on_left():
    wave_obj = sa.WaveObject.from_wave_file("left.wav")  # Replace 'left_sound.wav' with your left speaker sound file
    play_obj = wave_obj.play()
    play_obj.wait_done()

def play_sound_on_right():
    wave_obj = sa.WaveObject.from_wave_file("right.wav")  # Replace 'right_sound.wav' with your right speaker sound file
    play_obj = wave_obj.play()
    play_obj.wait_done()

def run(camera_url, window_name, sound_function):
    import cvlib as cv  # Import cvlib here
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(camera_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Detect objects and display image with bounding boxes
        bbox, label, conf = cv.detect_common_objects(im)
        im = draw_bbox(im, bbox, label, conf)
        cv2.imshow(window_name + " - Detection", im)

        # Print debug information
        print("Detected labels:", label)

        # Play sound based on camera detection
        if label and 'person' in label:  # Assuming you are interested in detecting persons
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
