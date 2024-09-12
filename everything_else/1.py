import cv2
import numpy as np
import requests
from urllib.request import urlopen

def load_video_stream(url):
    stream = urlopen(url)
    return stream

def main():
    # Replace these URLs with the actual URLs of your video streams
    url1 = 'http://192.168.4.1/'
    url2 = 'http://192.168.4.2/'

    stream1 = load_video_stream(url1)
    stream2 = load_video_stream(url2)

    cap1 = cv2.VideoCapture()
    cap1.open(url1)
    cap2 = cv2.VideoCapture()
    cap2.open(url2)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("Error reading video streams")
            break

        # Resize frames for a better display
        frame1 = cv2.resize(frame1, (640, 480))
        frame2 = cv2.resize(frame2, (640, 480))

        # Display frames side by side
        cv2.imshow('Video Stream 1', frame1)
        cv2.imshow('Video Stream 2', frame2)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
