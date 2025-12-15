import unittest
import numpy as np
from kalman_filter import KalmanFilter

class TestKalmanFilter(unittest.TestCase):
    def test_initialization_4d(self):
        kf = KalmanFilter(mode='4D')
        self.assertEqual(kf.x.shape, (4, 1))
        self.assertEqual(kf.F.shape, (4, 4))
        self.assertEqual(kf.H.shape, (2, 4))

    def test_initialization_6d(self):
        kf = KalmanFilter(mode='6D')
        self.assertEqual(kf.x.shape, (6, 1))
        self.assertEqual(kf.F.shape, (6, 6))
        self.assertEqual(kf.H.shape, (2, 6))

    def test_predict_update_cycle(self):
        kf = KalmanFilter(mode='4D', dt=1.0)
        # Initial state is 0
        kf.predict()
        self.assertTrue(np.allclose(kf.x, 0))
        
        # Update with measurement (10, 10)
        z = [10, 10]
        kf.update(z)
        
        # State should move towards measurement
        self.assertGreater(kf.x[0, 0], 0)
        self.assertGreater(kf.x[1, 0], 0)
        
        # Predict again
        kf.predict()
        # Should continue moving
        self.assertGreater(kf.x[0, 0], 0)

    def test_prediction_path(self):
        kf = KalmanFilter(mode='4D', dt=1.0)
        kf.x[2, 0] = 10 # vx = 10
        
        path = kf.get_predicted_path(t_pred=2.0, dt_step=1.0)
        # Should have 2 points
        self.assertEqual(len(path), 2)
        # First point should be roughly current + velocity
        # Note: get_predicted_path starts from current state and applies F
        # x_new = x + v*dt = 0 + 10*1 = 10
        self.assertEqual(path[0][0], 10)
        # Second point: 10 + 10 = 20
        self.assertEqual(path[1][0], 20)

if __name__ == '__main__':
    unittest.main()
