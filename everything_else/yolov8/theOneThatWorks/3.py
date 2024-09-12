import cv2
import urllib.request
import numpy as np
from cvlib.object_detection import detect_common_objects, draw_bbox
import multiprocessing

url1 = 'http://192.168.0.207/640x480.jpg' #stuck
url2 = 'http://192.168.0.154/640x480.jpg' #separate

def run(camera_url, window_name):
    import cvlib as cv  # Import cvlib here
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(camera_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Display original image
        # cv2.imshow(window_name + " - Original", im)

        # Detect objects and display image with bounding boxes
        bbox, label, conf = cv.detect_common_objects(im)
        im = draw_bbox(im, bbox, label, conf)
        cv2.imshow(window_name + " - Detection", im)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("started")
    try:
        process1 = multiprocessing.Process(target=run, args=(url1, "Camera 1"))
        process2 = multiprocessing.Process(target=run, args=(url2, "Camera 2"))

        process1.start()
        process2.start()

        process1.join()
        process2.join()
    except KeyboardInterrupt:
        pass