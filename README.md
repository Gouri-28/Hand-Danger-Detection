# Real-Time Hand Tracking & Danger Boundary Detection

## Objective

A real-time computer vision prototype that detects hand position and warns when it approaches a virtual danger zone. Built using classical CV techniques without MediaPipe or OpenPose.

## Features

- Contour-based real-time hand tracking using HSV segmentation
- Virtual boundary drawn on screen
- SAFE / WARNING / DANGER states
- "DANGER DANGER" alert displayed visually
- Runs above 25 FPS on CPU

## How to Run

pip install -r requirements.txt
python hand_danger_poc.py
