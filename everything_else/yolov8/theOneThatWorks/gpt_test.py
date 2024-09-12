import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def get_frames(url):
    while True:
        try:
            r = requests.get(url, stream=True, timeout=10)
            print(f"Status Code: {r.status_code}")
            if r.status_code == 200:
                bytes = b''
                for chunk in r.iter_content(chunk_size=1024):
                    bytes += chunk
                    a = bytes.find(b'\xff\xd8')  # Start of JPEG
                    b = bytes.find(b'\xff\xd9')  # End of JPEG
                    if a != -1 and b != -1:
                        jpg = bytes[a:b+2]
                        bytes = bytes[b+2:]
                        # Use Pillow to open the image and convert to numpy array
                        try:
                            img = Image.open(BytesIO(jpg))
                            frame = np.array(img)
                            cv2.imshow('Frame', frame)
                            if cv2.waitKey(1) == ord('q'):
                                return
                        except Exception as e:
                            print(f"Error decoding frame: {e}")
                    elif b != -1:
                        bytes = b''
            else:
                print(f"Failed to connect to stream. Status code: {r.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the stream: {e}")
            print("Attempting to reconnect...")
            cv2.waitKey(1000)

stream_url = 'http://192.168.0.154:81/stream'
cv2.namedWindow('Frame', cv2.WINDOW_AUTOSIZE)
get_frames(stream_url)
cv2.destroyAllWindows()
