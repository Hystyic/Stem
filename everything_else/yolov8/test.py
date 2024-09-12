from ultralytics import YOLO
# from ultralytics.yolo.v8.detect.predict import DetectionPredictor
import cv2



model = YOLO("yolov8x.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.predict(source="0", show=True)  # predict on an image
print(results)