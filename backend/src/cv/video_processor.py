import cv2
import numpy as np
from logic.scoring_engine import PadelScoringEngine
from cv.calibration import CameraCalibration

class PadelVideoProcessor:
    def __init__(self, camera_config):
        self.engine = PadelScoringEngine()
        self.calib = CameraCalibration()
        # Setup the 3D projection matrix from the user's UI calibration
        self.P = self.calib.calculate_projection_matrix(
            pos_x=camera_config['pos_x'],
            pos_y=camera_config['pos_y'],
            height=camera_config['height'],
            tilt_deg=camera_config['tilt'],
            pan_deg=0
        )
        self.ball_prev_y = None
        self.events = []

    def analyze_frame(self, frame_data, timestamp):
        """
        frame_data: { "ball": [x,y,w,h], "players": [...] }
        timestamp: current video time in seconds
        """
        if not frame_data["ball"]:
            return

        # 1. Get current 2D ball position
        ball_x = frame_data["ball"][0]
        ball_y = frame_data["ball"][1]

        # 2. Logic: Detect Bounce (Velocity Flip)
        if self.ball_prev_y is not None:
            dy = ball_y - self.ball_prev_y
            if dy > 0 and self.ball_prev_y < 0: # Check for direction change
                 # Detect which court it bounced in (meters)
                 real_x, real_y = self.calib.world_to_pixel(ball_x, ball_y, 0, self.P)
                 self.log_event("Bounce!", timestamp)

        self.ball_prev_y = ball_y

    def log_event(self, label, time):
        # Determine current score from the engine
        score_display = self.engine.get_score_display()
        self.events.append({
            "id": len(self.events) + 1,
            "time": round(time, 1),
            "label": label,
            "score": score_display["score"]
        })
        return self.events
