import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional, List, Dict
from cv.court_calibration import CourtCalibration

BALL_DIAMETER_M = 0.065
LOST_THRESHOLD_FRAMES = 60


class BallTracker:
    def __init__(self, calibration: CourtCalibration, fps: float = 30.0):
        self.calibration = calibration
        self.fps = fps
        self.dt = 1.0 / fps
        self.trajectory: List[Dict] = []
        self.is_lost = False
        self._miss_count = 0
        self._initialized = False
        self._prev_court_pos = None

        self._kf = KalmanFilter(dim_x=4, dim_z=2)
        self._kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self._kf.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self._kf.P *= 100
        self._kf.R = np.eye(2) * 5
        self._kf.Q = np.eye(4) * 0.1
        self._ground_ball_size: Optional[float] = 30.0  # Default ground ball size in pixels

    def update(self, bbox: Optional[List[float]], frame_number: int) -> Optional[Dict]:
        timestamp = frame_number * self.dt
        if bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            court_x, court_y = self.calibration.pixel_to_court(cx, cy)
            z = self._estimate_z(bbox_size, court_x, court_y)
            if not self._initialized:
                self._kf.x = np.array([court_x, court_y, 0, 0])
                self._initialized = True
            else:
                self._kf.predict()
                self._kf.update(np.array([court_x, court_y]))
            self._miss_count = 0
            self.is_lost = False
            speed = self._compute_speed(court_x, court_y)
            self._prev_court_pos = (court_x, court_y)
            pos = {"x": float(self._kf.x[0]), "y": float(self._kf.x[1]),
                   "z": z, "speed": speed, "timestamp": timestamp,
                   "frame": frame_number, "detected": True}
            self.trajectory.append(pos)
            return pos
        else:
            if not self._initialized:
                return None
            self._miss_count += 1
            if self._miss_count >= LOST_THRESHOLD_FRAMES:
                self.is_lost = True
                return None
            self._kf.predict()
            court_x = float(self._kf.x[0])
            court_y = float(self._kf.x[1])
            speed = self._compute_speed(court_x, court_y)
            self._prev_court_pos = (court_x, court_y)
            pos = {"x": court_x, "y": court_y, "z": 0.0, "speed": speed,
                   "timestamp": timestamp, "frame": frame_number, "detected": False}
            self.trajectory.append(pos)
            return pos

    def _estimate_z(self, bbox_size: float, court_x: float, court_y: float) -> float:
        # Update ground ball size to track minimum detected size
        if self._ground_ball_size is None or bbox_size < self._ground_ball_size:
            self._ground_ball_size = bbox_size

        if self._ground_ball_size <= 0:
            return 0.0
        ratio = bbox_size / self._ground_ball_size
        z = max(0.0, (ratio - 1.0) * 3.0)
        return round(z, 2)

    def _compute_speed(self, x: float, y: float) -> float:
        if self._prev_court_pos is None:
            return 0.0
        dx = x - self._prev_court_pos[0]
        dy = y - self._prev_court_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        speed_ms = dist / self.dt
        speed_kmh = speed_ms * 3.6
        return round(speed_kmh, 1)
