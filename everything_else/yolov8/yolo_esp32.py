import cv2
import requests
from ultralytics import YOLO

model = YOLO("yolov8x.pt")

cap1 = cv2.VideoCapture("http://192.168.221.91/stream")
cap2 = cv2.VideoCapture("http://192.168.221.152/stream")

while True:

    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    results1 = model.predict(frame1)
    results2 = model.predict(frame2)

    model.render(frame1)
    model.render(frame2)

    cv2.imshow("Frame 1", frame1)
    cv2.imshow("Frame 2", frame2)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap1.release()
cap2.release()

cv2.destroyAllWindows()