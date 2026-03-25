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

            # Estimate ball height:
            # Method 1: Known ball size + focal length → distance → height (if real bbox)
            # Method 2: Pixel vertical velocity accumulator (for TrackNet's fixed bbox)
            z = self._estimate_z_from_depth(cx, cy, bbox_size)
            if z < 0.05 and hasattr(self.calibration, 'project_to_ground'):
                z = self._estimate_z_from_camera(cx, cy)

            # Project to court coordinates
            if hasattr(self.calibration, 'project_to_height') and z > 0.2:
                court_x, court_y = self.calibration.project_to_height(cx, cy, z)
            elif hasattr(self.calibration, 'project_to_ground'):
                court_x, court_y = self.calibration.project_to_ground(cx, cy)
            else:
                court_x, court_y = self.calibration.pixel_to_court(cx, cy)
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

    BALL_REAL_DIAMETER = 0.065  # padel ball = 6.5cm

    def _estimate_z_from_depth(self, px: float, py: float, bbox_size: float) -> float:
        """Estimate ball height using known ball size + camera focal length.

        distance_from_camera = (focal_length × real_diameter) / pixel_diameter
        Then: 3D position = camera_pos + distance × ray_direction
        Height = 3D_position.z
        """
        try:
            if bbox_size < 5 or bbox_size > 100:
                return 0.0  # unreasonable size, skip

            # Skip if bbox is the fake TrackNet fixed size (always ~20px)
            if hasattr(self, '_bbox_sizes'):
                self._bbox_sizes.append(bbox_size)
                if len(self._bbox_sizes) > 30:
                    self._bbox_sizes.pop(0)
                # If all sizes are identical (±1px), it's TrackNet fake bbox
                if len(self._bbox_sizes) > 10:
                    unique = len(set(int(s) for s in self._bbox_sizes))
                    if unique <= 2:  # all same size = fake bbox
                        return 0.0
            else:
                self._bbox_sizes = [bbox_size]

            # Need camera matrix for focal length
            if not hasattr(self.calibration, 'camera_matrix') or self.calibration.camera_matrix is None:
                return 0.0

            focal_length = self.calibration.camera_matrix[0, 0]
            if focal_length <= 0:
                return 0.0

            # Distance from camera
            distance = (focal_length * self.BALL_REAL_DIAMETER) / bbox_size

            # Get camera position and ray direction
            import cv2 as cv
            R, _ = cv.Rodrigues(self.calibration.rvec)
            cam_pos = (-R.T @ self.calibration.tvec.flatten())

            # Ray from camera through pixel
            px_norm = np.array([
                (px - self.calibration.camera_matrix[0, 2]) / focal_length,
                (py - self.calibration.camera_matrix[1, 2]) / self.calibration.camera_matrix[1, 1],
                1.0,
            ])
            ray_dir = R.T @ px_norm
            ray_dir = ray_dir / np.linalg.norm(ray_dir)

            # 3D ball position
            ball_3d = cam_pos + distance * ray_dir

            # Height is the Z component (Z is up in our court coordinate system)
            height = float(ball_3d[2])
            return round(max(0.0, min(height, 5.0)), 2)
        except Exception:
            return 0.0

    def _estimate_z_from_camera(self, px: float, py: float) -> float:
        """Estimate ball height by fitting parabola vs line to recent pixel-Y trajectory.

        Key insight: airborne ball follows a parabola in pixel Y (due to gravity).
        Ground ball follows ~linear pixel Y (just perspective change).
        By comparing parabola vs line fit over last N frames, we can tell if
        the ball is in the air and estimate its height from the parabola shape.
        """
        try:
            if not hasattr(self, '_pixel_history'):
                self._pixel_history = []

            self._pixel_history.append((px, py))
            if len(self._pixel_history) > 20:
                self._pixel_history.pop(0)

            # Need at least 6 frames to fit a meaningful curve
            if len(self._pixel_history) < 6:
                return 0.0

            # Extract pixel Y values for last N frames
            ys = np.array([p[1] for p in self._pixel_history])
            xs = np.arange(len(ys), dtype=np.float64)

            # Fit line: y = ax + b
            line_coeffs = np.polyfit(xs, ys, 1)
            line_fit = np.polyval(line_coeffs, xs)
            line_error = np.mean((ys - line_fit) ** 2)

            # Fit parabola: y = ax² + bx + c
            para_coeffs = np.polyfit(xs, ys, 2)
            para_fit = np.polyval(para_coeffs, xs)
            para_error = np.mean((ys - para_fit) ** 2)

            # Parabola "a" coefficient: positive = opening upward in pixel space
            # = ball went UP then came DOWN (since pixel Y decreases when ball rises)
            a = para_coeffs[0]

            # Is parabola a significantly better fit than line?
            # And does it have the right shape (a > 0 = ball arc)?
            is_airborne = (para_error < line_error * 0.7) and (a > 0.1)

            if not is_airborne:
                return 0.0

            # Estimate height from parabola
            # The parabola vertex is at x = -b/(2a), y_vertex = c - b²/(4a)
            vertex_x = -para_coeffs[1] / (2 * para_coeffs[0])
            vertex_y = np.polyval(para_coeffs, vertex_x)

            # Current pixel Y vs vertex Y = pixel displacement from peak
            # Convert to meters using camera model
            current_y = ys[-1]
            peak_displacement = abs(vertex_y - np.mean(ys))  # pixel distance from mean

            # Get pixels-per-meter at this depth
            ground_x, ground_y = self.calibration.project_to_ground(px, py)
            try:
                gnd_px, gnd_py = self.calibration.court_to_pixel(ground_x, ground_y, 0.0)
                elv_px, elv_py = self.calibration.court_to_pixel(ground_x, ground_y, 1.0)
                ppm = abs(gnd_py - elv_py)
            except Exception:
                ppm = 50.0

            if ppm < 1:
                ppm = 50.0

            # Height = how far the current pixel Y deviates from the line fit
            # (parabola deviation from linear = the "airborne" component)
            deviation = abs(current_y - np.polyval(line_coeffs, xs[-1]))
            height = deviation / ppm

            return round(max(0.0, min(height, 5.0)), 2)
        except Exception:
            return 0.0

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
