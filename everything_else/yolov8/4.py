import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO
import multiprocessing


url1 = 'http://192.168.221.91/cam-hi.jpg'
url2 = 'http://192.168.221.152/cam-hi.jpg'

def draw_boxes(image, results):
    try:
        for det in results.xyxy[0]:
            bbox = det[:4].cpu().numpy()
            label = int(det[5])
            conf = float(det[4])
            bbox = bbox.astype(int)
            image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            image = cv2.putText(image, f'{model.names[label]} {conf:.2f}', (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    except AttributeError:
        pass  # Handle the case when xyxy attribute is not present

    return image

def run(camera_url, window_name):
    model = YOLO("yolov8x.pt")
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(camera_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Detect objects and draw bounding boxes
        results = model(im)
        im_with_boxes = draw_boxes(im.copy(), results)

        cv2.imshow(window_name, im_with_boxes)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("started")
    try:
        process1 = multiprocessing.Process(target=run, args=(url1, "Detection Camera 1"))
        process2 = multiprocessing.Process(target=run, args=(url2, "Detection Camera 2"))

        process1.start()
        process2.start()

        process1.join()
        process2.join()
    except KeyboardInterrupt:
        pass
