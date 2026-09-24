# Volume Control

Hand Gesture-based Volume Control using MediaPipe and OpenCV

## Description

This project uses Computer Vision to detect hand gestures and control your system volume. It uses:
- **MediaPipe** for hand detection and tracking
- **OpenCV** for image processing and display
- **pycaw** for Windows audio control

## Installation

```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install opencv-python mediapipe numpy comtypes pycaw
```

## Usage

Run the application:
```bash
python volume_control.py
```

Press **'q'** to quit the application.

## How it Works

The application detects your hand using MediaPipe and measures the distance between your thumb and index finger. This distance is mapped to your system volume level:
- Small distance (fingers close) = Low volume
- Large distance (fingers apart) = High volume
