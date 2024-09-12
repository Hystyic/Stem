import cv2

build_info = cv2.getBuildInformation()
if "CUDA" in build_info:
    print("OpenCV was built with CUDA support.")
else:
    print("OpenCV was not built with CUDA support.")
