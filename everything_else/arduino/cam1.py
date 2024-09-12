# import cv2
# import numpy as np
# import urllib.request
# import concurrent.futures

# url = 'http://192.168.98.152/cam-hi.jpg'
# im = None

# def detect_objects(image):
#     net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

#     with open('coco.names', 'r') as f:
#         classes = f.read().strip().split('\n')

#     height, width = image.shape[:2]
#     blob = cv2.dnn.blobFromImage(image, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
#     net.setInput(blob)
#     output_layers = net.getUnconnectedOutLayersNames()
#     outs = net.forward(output_layers)

#     class_ids = []
#     confidences = []
#     boxes = []

#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5:
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)

#                 class_ids.append(class_id)
#                 confidences.append(float(confidence))
#                 boxes.append([x, y, w, h])

#     indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

#     for i in indices:
#         i = i[0]
#         box = boxes[i]
#         x, y, w, h = box
#         label = classes[class_ids[i]]
#         confidence = confidences[i]
#         color = (255, 0, 0)
#         cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
#         cv2.putText(image, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     return image

# def run1():
#     cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
#     while True:
#         img_resp = urllib.request.urlopen(url)
#         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         im = cv2.imdecode(imgnp, -1)

#         cv2.imshow('live transmission', im)
#         key = cv2.waitKey(5)
#         if key == ord('q'):
#             break

#     cv2.destroyAllWindows()

# def run2():
#     cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
#     while True:
#         img_resp = urllib.request.urlopen(url)
#         imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
#         im = cv2.imdecode(imgnp, -1)

#         detected_image = detect_objects(im)
#         cv2.imshow('detection', detected_image)

#         key = cv2.waitKey(5)
#         if key == ord('q'):
#             break

#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     print("started")
#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         f1 = executor.submit(run1)
#         f2 = executor.submit(run2)

import cv2
import numpy as np
import urllib.request
import concurrent.futures

url = 'http://192.168.98.152/cam-hi.jpg'
im = None

# 

def detect_objects(image):
    net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')

    classes = []
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    height, width, channels = image.shape

    blob = cv2.dnn.blobFromImage(image, scalefactor=0.00392, size=(850, 1000), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                print("123")

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = (0, 255, 0)
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f'{label} {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            print("456")

    return image


def run1():
    cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        cv2.imshow('live transmission', im)
        key = cv2.waitKey(5)
        if key == ord('q'):
            break
        print("789")

    cv2.destroyAllWindows()

def run2():
    cv2.namedWindow("detection", cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        detected_image = detect_objects(im)
        cv2.imshow('detection', detected_image)

        key = cv2.waitKey(5)
        if key == ord('q'):
            break

        print("9090")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print("started")
    with concurrent.futures.ProcessPoolExecutor() as executor:
        f1 = executor.submit(run1)
        f2 = executor.submit(run2)
