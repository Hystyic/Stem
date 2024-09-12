import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa
from cvlib.object_detection import detect_common_objects

url1 = 'http://192.168.0.207/640x480.jpg'  # Camera 1 URL
url2 = 'http://192.168.0.154/640x480.jpg'  # Camera 2 URL

def play_sound_on_left():
    wave_obj = sa.WaveObject.from_wave_file("left.wav")  # Replace with your left speaker sound file
    play_obj = wave_obj.play()
    play_obj.wait_done()

def play_sound_on_right():
    wave_obj = sa.WaveObject.from_wave_file("right.wav")  # Replace with your right speaker sound file
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Define average sizes for different objects in meters
AVERAGE_SIZES = {
    'person': 0.5,   # Average width of a person
    'phone': 0.08,   # Average length of a phone
    'laptop': 0.35,  # Average width of a laptop
    'chair': 0.45,   # Average width of a chair
    'table': 1.2,    # Average length of a table
    # Add more objects as needed
}

CALIBRATED_FOCAL_LENGTH = 500  # Example focal length, needs calibration

def run(camera_url, window_name, sound_function):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(camera_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Detect objects
        bbox, label, conf = detect_common_objects(im)

        # Draw bounding boxes and labels with distances
        for box, lbl in zip(bbox, label):
            x1, y1, x2, y2 = box
            object_width_in_pixels = x2 - x1

            # Select the average size based on the object label, default to person if not found
            object_size = AVERAGE_SIZES.get(lbl, AVERAGE_SIZES['person'])

            distance = (object_size * CALIBRATED_FOCAL_LENGTH) / object_width_in_pixels
            label_with_distance = f"{lbl}: {distance:.2f} meters"
            cv2.putText(im, label_with_distance, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow(window_name + " - Detection", im)

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
