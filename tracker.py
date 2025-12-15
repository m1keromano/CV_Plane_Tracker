from kalman_filter import KalmanFilter
import config
import numpy as np

class PlaneTracker:
    def __init__(self, fps):
        self.fps = fps
        self.dt_frame = 1.0 / fps
        self.dt_update = self.dt_frame * config.MEASUREMENT_SKIP
        
        self.kf = KalmanFilter(
            mode=config.STATE_VECTOR_MODE,
            dt=self.dt_update,
            p0_var=config.P0_VAR,
            r_var=config.R_VAR,
            q_var=config.Q_VAR
        )
        
        self.history = [] # List of (x, y) tuples of corrected positions
        self.is_initialized = False

    def update(self, measurement):
        """
        Perform a predict/update cycle.
        measurement: (x, y) tuple or None if no detection.
        """
        # If not initialized and we have a measurement, initialize state
        if not self.is_initialized:
            if measurement is not None:
                self.kf.x[0, 0] = measurement[0]
                self.kf.x[1, 0] = measurement[1]
                self.is_initialized = True
                self.history.append(measurement)
            return

        # Predict
        self.kf.predict()

        # Update if we have a measurement
        if measurement is not None:
            self.kf.update(measurement)
            
        # Store corrected position
        # Note: In a real system, if we miss a measurement, we might store the predicted position.
        # Here we store the state's position (which is the predicted one if no update occurred, 
        # or the corrected one if update occurred).
        pos = (int(self.kf.x[0, 0]), int(self.kf.x[1, 0]))
        self.history.append(pos)

    def get_prediction_path(self):
        """
        Get the future path based on current state.
        """
        if not self.is_initialized:
            return []
        return self.kf.get_predicted_path(config.T_PRED, self.dt_frame)
