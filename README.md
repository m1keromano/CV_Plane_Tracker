# CV Plane Tracker

A robust computer vision system for tracking planes in ground-to-air video using YOLO and a Kalman Filter.

## Features
- **Object Detection**: Uses YOLOv8 for robust plane detection.
- **Kalman Filter**: 
    - **4D Mode**: Constant Velocity model ($x, y, \dot{x}, \dot{y}$).
    - **6D Mode**: Constant Acceleration model ($x, y, \dot{x}, \dot{y}, \ddot{x}, \ddot{y}$), recommended for maneuvering aircraft.
- **Visualization**:
    - **Blue Path**: Historical tracked path.
    - **Red Path**: Predicted future path.

## Installation

```bash
pip install opencv-python ultralytics numpy
```

## Usage

1.  **Configuration**: Edit `config.py` to adjust parameters:
    - `MEASUREMENT_SKIP`: Number of frames to skip between updates.
    - `T_PRED`: Prediction time in seconds.
    - `STATE_VECTOR_MODE`: '4D' or '6D'.

2.  **Run**:
    ```bash
    python main.py <path_to_video>
    ```

## Structure
- `main.py`: Entry point, video processing loop, and visualization.
- `tracker.py`: Manages the Kalman Filter and tracking logic.
- `kalman_filter.py`: Core Kalman Filter implementation.
- `config.py`: Configuration constants.