# Stem - Bone Conducting Headset for Blind Assistance

**Stem** is a bone-conducting headset equipped with two onboard cameras that provides haptic and audio assistance for visually impaired individuals. By using real-time object detection and spatial audio cues, Stem helps users navigate their surroundings more effectively. The headset processes visual data through the onboard cameras, detects objects, and provides spatial audio feedback using bone conduction technology.

## Features

- **Bone Conduction Technology**: Allows users to hear spatial audio cues without blocking their ears, ensuring awareness of environmental sounds.
- **Dual Camera System**: Two onboard cameras capture the environment, enabling stereo vision and object detection.
- **Object Detection with YOLOv8**: Real-time object detection is performed using the YOLOv8 model, allowing the system to identify various objects in the environment.
- **Spatial Audio Feedback**: Audio cues are played through bone conduction based on the position and size of detected objects, with the frequency and rate of sound dynamically changing depending on the proximity and size of the objects.
- **Stereo Vision for Depth Perception**: The system uses stereo vision to compute depth information, helping users understand the distance of detected objects.
- **Haptic Feedback (Optional)**: Spatial audio cues are complemented by haptic feedback, providing an extra layer of guidance for users.

## How It Works

1. **Camera Input**: The two cameras capture live video streams from the surroundings.
2. **Object Detection**: The video frames are processed using the YOLOv8 model to detect objects like people, cars, bicycles, and more.
3. **Stereo Vision**: Depth perception is computed using stereo vision techniques to determine how close objects are to the user.
4. **Audio Feedback**: Based on the size and distance of the detected objects, spatial audio cues are generated, helping the user understand the object’s proximity and location (left or right).

### Object Detection Classes

Stem can detect a wide range of objects, including:

- People
- Cars
- Bicycles
- Dogs
- Traffic signs
- Many more...

A full list of object classes can be found in the code.

## Installation and Setup

### Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- NumPy (`pip install numpy`)
- Pygame (`pip install pygame`)
- Ultralytics YOLO (`pip install ultralytics`)

### Steps to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Hystyic/Stem.git
   cd stem
   pip install -r requirements.txt

2. **Flash ESP firmware to ESP32CAM**
3. **Run the application**
    ```bash
   python stem.py

