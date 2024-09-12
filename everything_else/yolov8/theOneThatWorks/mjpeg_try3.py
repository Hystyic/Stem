import cv2
import numpy as np
import urllib.request
import time

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# URL of the MJPEG stream
stream_url = 'http://192.168.0.207/640x480.mjpeg'

# Function to fetch a frame from the MJPEG stream
def fetch_frame(stream_url):
    while True:
        try:
            with urllib.request.urlopen(stream_url) as stream:
                bytes = b''
                while True:
                    bytes += stream.read(1024)
                    a = bytes.find(b'\xff\xd8')  # Start of JPEG
                    b = bytes.find(b'\xff\xd9')  # End of JPEG
                    if a != -1 and b != -1:
                        jpg = bytes[a:b+2]
                        bytes = bytes[b+2:]
                        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                        return frame
        except ConnectionResetError:
            print("Connection was reset. Reconnecting...")
            time.sleep(1)  # Pause briefly before reconnecting

# Main loop
while True:
    frame = fetch_frame(stream_url)
    if frame is not None:
        # Face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame
        cv2.imshow("Haar Cascade Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
