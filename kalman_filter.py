import numpy as np

class KalmanFilter:
    def __init__(self, mode='6D', dt=1.0, p0_var=1000.0, r_var=100.0, q_var=0.1):
        """
        Initialize the Kalman Filter.
        
        Args:
            mode (str): '4D' for Constant Velocity or '6D' for Constant Acceleration.
            dt (float): Time step between updates.
            p0_var (float): Variance for Initial State Covariance P0.
            r_var (float): Variance for Measurement Noise Covariance R.
            q_var (float): Variance for Process Noise Covariance Q.
        """
        self.mode = mode
        self.dt = dt
        
        if mode == '4D':
            # State: [x, y, vx, vy]
            self.n_dim = 4
            self.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
            # Measurement matrix: we only measure x, y
            self.H = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ])
        elif mode == '6D':
            # State: [x, y, vx, vy, ax, ay]
            self.n_dim = 6
            dt2 = 0.5 * dt**2
            self.F = np.array([
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            # Measurement matrix: we only measure x, y
            self.H = np.array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0]
            ])
        else:
            raise ValueError("Invalid mode. Choose '4D' or '6D'.")

        # Initial State Vector
        self.x = np.zeros((self.n_dim, 1))
        
        # Initial State Covariance Matrix P
        self.P = np.eye(self.n_dim) * p0_var
        
        # Measurement Noise Covariance Matrix R (2x2 for x, y)
        self.R = np.eye(2) * r_var
        
        # Process Noise Covariance Matrix Q
        # Using a simplified Q for now, can be more sophisticated (e.g., discrete white noise model)
        self.Q = np.eye(self.n_dim) * q_var

    def predict(self, dt=None):
        """
        Predict the next state.
        Allows overriding dt for variable time steps if needed, 
        though usually F is constant if dt is constant.
        If dt changes, F needs to be rebuilt.
        """
        if dt is not None and dt != self.dt:
            self._update_F(dt)
            
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        """
        Update the state with a new measurement z = [x, y].
        """
        z = np.array(z).reshape(2, 1)
        
        # Innovation
        y = z - self.H @ self.x
        
        # Innovation Covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update State
        self.x = self.x + K @ y
        
        # Update Covariance
        I = np.eye(self.n_dim)
        self.P = (I - K @ self.H) @ self.P
        
        return self.x

    def _update_F(self, dt):
        """Helper to update transition matrix F for a new dt."""
        self.dt = dt
        if self.mode == '4D':
            self.F = np.array([
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        elif self.mode == '6D':
            dt2 = 0.5 * dt**2
            self.F = np.array([
                [1, 0, dt, 0, dt2, 0],
                [0, 1, 0, dt, 0, dt2],
                [0, 0, 1, 0, dt, 0],
                [0, 0, 0, 1, 0, dt],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])

    def get_predicted_path(self, t_pred, dt_step):
        """
        Generate a predicted path for t_pred seconds into the future.
        Returns a list of (x, y) points.
        Does NOT modify the current state.
        """
        path = []
        
        # Create a temporary F for the small time step
        if self.mode == '4D':
            F_step = np.array([
                [1, 0, dt_step, 0],
                [0, 1, 0, dt_step],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        else: # 6D
            dt2 = 0.5 * dt_step**2
            F_step = np.array([
                [1, 0, dt_step, 0, dt2, 0],
                [0, 1, 0, dt_step, 0, dt2],
                [0, 0, 1, 0, dt_step, 0],
                [0, 0, 0, 1, 0, dt_step],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1]
            ])
            
        x_curr = self.x.copy()
        
        steps = int(t_pred / dt_step)
        for _ in range(steps):
            x_curr = F_step @ x_curr
            path.append((int(x_curr[0, 0]), int(x_curr[1, 0])))
            
        return path
