
# Configuration for CV Plane Tracker

# Video Processing
MEASUREMENT_SKIP = 15  # Process every Nth frame
RESIZE_WIDTH = 1280   # Resize video for consistent processing (optional)

# Prediction
T_PRED = 3.0          # Prediction horizon in seconds

# Kalman Filter
# Options: '4D' (Constant Velocity) or '6D' (Constant Acceleration)
STATE_VECTOR_MODE = '6D' 

# Initial Covariance Values
P0_VAR = 100.0        # Initial State Covariance
R_VAR = 1.0           # Measurement Noise (Lower = trust measurement more)
Q_VAR = 5.0           # Process Noise (Higher = trust model dynamics less / allow more change)
