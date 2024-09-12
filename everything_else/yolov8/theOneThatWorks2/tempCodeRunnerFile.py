import cv2
import urllib.request
import numpy as np
from cvlib.object_detection import detect_common_objects, draw_bbox
import multiprocessing
import simpleaudio as sa
from ultralytics import YOLO
import torch


torch.cuda.set_device(0)



model = YOLO("yolov8n.pt")
model.to('cuda')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


url1 = 'http://192.168.0.142/640x480.jpg'  
url2 = 'http://192.168.0.142/640x480.jpg'  

def is_stream_available(stream_url):
  try:
    response = urllib.request.urlopen(stream_url)
    print("func")
    if response.status == 200:
      return True
    else:
      return False
  except urllib.request.URLError:
    return False


def run(camera_url, window_name):
  cap = cv2.VideoCapture(camera_url)
  print("func2")

  while cap.isOpened():
    ret, frame = cap.read()

    if ret:
      
      frame = cv2.resize(frame, (640, 640))  
      
      frame = np.transpose(frame, (2, 0, 1)) / 255.0
      frame = np.expand_dims(frame, axis=0)
      
      frame = torch.from_numpy(frame).float()

      frame = frame.to(device)
      
      detections = model(frame)
      
      for detection in detections:
        cv2.rectangle(frame, (detection.xmin, detection.ymin), (detection.xmax, detection.ymax), (0, 0, 255), 2)

      
      cv2.imshow(window_name, frame)

      
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    else:
      break

  
  cap.release()

  
  cv2.destroyAllWindows()


if is_stream_available(url1) and is_stream_available(url2):
  print("both streams avail")
  
  if __name__ == '__main__':
    process1 = multiprocessing.Process(target=run, args=(url1, "Camera 1"))
    print("process1 starts")
    process2 = multiprocessing.Process(target=run, args=(url2, "Camera 2"))
    print("process2 starts")
    process1.start()
    process2.start()

    process1.join()
    process2.join()
else:
  
  print("One or both of the video streams are not available.")
