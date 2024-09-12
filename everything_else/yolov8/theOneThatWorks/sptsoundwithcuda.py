import cv2
import urllib.request
import numpy as np
import multiprocessing
import simpleaudio as sa

# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")  # Replace with the path to your YOLO files
classes = []

with open("coco.names", "r") as f:  # Replace with the path to your COCO names file
    classes = f.read().strip().split("\n")

url1 = 'http://192.168.0.142/640x480.jpg'  # stuck
url2 = 'http://192.168.0.154/640x480.jpg'  # separate

def play_sound_on_left():
    wave_obj = sa.WaveObject.from_wave_file("left.wav")  # Replace 'left_sound.wav' with your left speaker sound file
    play_obj = wave_obj.play()
    play_obj.wait_done()

def play_sound_on_right():
    wave_obj = sa.WaveObject.from_wave_file("right.wav")  # Replace 'right_sound.wav' with your right speaker sound file
    play_obj = wave_obj.play()
    play_obj.wait_done()

def run(camera_url, window_name, sound_function):
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    while True:
        img_resp = urllib.request.urlopen(camera_url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        im = cv2.imdecode(imgnp, -1)

        # Use YOLO for object detection
        blob = cv2.dnn.blobFromImage(im, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward()

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * im.shape[1])
                    center_y = int(detection[1] * im.shape[0])
                    w = int(detection[2] * im.shape[1])
                    h = int(detection[3] * im.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        # Draw bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indices:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(im, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(window_name + " - Detection", im)

        # Print debug information
        print("Detected labels:", classes[class_id])

        # Play sound based on camera detection
        if classes and 'person' in classes[class_id]:  # Assuming you are interested in detecting persons
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
