import torch
import cv2
import numpy as np
from ultralytics import yolo

# Load the model
model = yolo.load_model('yolov5s.pt')

# Read the input image
img = cv2.imread('image.jpg')

# Convert the image to blob
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)

# Set the input blob
model.setInput(blob)

# Get the names of the output layers
outNames = model.getUnconnectedOutLayersNames()

# Run the forward pass
detections = model.forward(outNames)

# Loop over the detections
for detection in detections:
    # Get the confidence score
    confidence = detection[5]

    # Filter out weak detections
    if confidence > 0.5:
        # Get the bounding box coordinates
        box = detection[0:4] * np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        (startX, startY, endX, endY) = box.astype("int")

        # Draw the bounding box on the image
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)

# Display the output image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()