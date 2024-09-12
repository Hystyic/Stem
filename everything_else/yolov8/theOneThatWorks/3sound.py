import pygame
import cv2
import urllib.request
import numpy as np
from cvlib.object_detection import detect_common_objects, draw_bbox
import multiprocessing
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  tf.config.experimental.set_memory_growth(gpus[0], True)

pygame.mixer.init()
url1 = 'http://192.168.96.152/640x480.jpg'
url2 = 'http://192.168.96.91/640x480.jpg'

left_sound = pygame.mixer.Sound('left.mp3')
right_sound = pygame.mixer.Sound('right.mp3')

def set_volume(sound, confidence):
  volume = max(0.0, min(1.0, confidence))
  sound.set_volume(volume)

def run(camera_url, window_name, channel):
  cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
  while True:
    img_resp = urllib.request.urlopen(camera_url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    im = cv2.imdecode(imgnp, -1)

    bbox, label, conf = detect_common_objects(im)
    im = draw_bbox(im, bbox, label, conf)
    cv2.imshow(window_name + " - Detection", im)

    if bbox:
      max_confidence = max(conf) if conf else 0

      channel = pygame.mixer.find_channel(True)

      if (channel == 'left' and not left_sound.get_busy()) or \
          (channel == 'right' and not right_sound.get_busy()):
        if channel == 'left':
          set_volume(left_sound, max_confidence)
          left_sound.play()
        elif channel == 'right':
          set_volume(right_sound, max_confidence)
          right_sound.play()

    key = cv2.waitKey(5)
    if key == ord('q'):
      break

  cv2.destroyAllWindows()

left_channel_sound = 'left.mp3'  
right_channel_sound = 'right.mp3'

if __name__ == '__main__':
  print("started")
  try:
    process1 = multiprocessing.Process(target=run, args=(url1, "Camera 1", 'left'))
    process2 = multiprocessing.Process(target=run, args=(url2, "Camera 2", 'right'))

    process1.start()
    process2.start()

    process1.join()
    process2.join()
  except KeyboardInterrupt:
    process1.terminate()
    process2.terminate()
    pygame.quit()
    cv2.destroyAllWindows()
    print("Stopped")