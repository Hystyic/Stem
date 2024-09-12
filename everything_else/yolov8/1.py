import cv2
import urllib.request
import numpy as np

url = "http://192.168.4.1/"  # Replace with the actual IP address of your ESP32-CAM

cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error reading frame")
        break

    cv2.imshow("ESP32-CAM Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
