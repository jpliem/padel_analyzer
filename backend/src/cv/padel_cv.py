import cv2
import torch
from ultralytics import YOLO
import numpy as np

class PadelCV:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLO model
        self.model = YOLO(model_path)
        self.ball_trajectory = []
        self.players = {} # ID: (x, y)
        self.court_homography = None

    def process_frame(self, frame):
        """
        Processes a single frame:
        1. Detects objects (ball, players).
        2. Updates trajectories.
        3. Returns annotated frame and data.
        """
        results = self.model(frame)
        detections = results[0].boxes
        
        frame_data = {
            "ball": None,
            "players": []
        }

        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            if cls == 0: # Person (Player)
                frame_data["players"].append(xyxy)
            elif cls == 32: # Sports ball (Ball) - generic YOLO class
                frame_data["ball"] = xyxy
                self.ball_trajectory.append(xyxy)

        return frame, frame_data

    def calibrate_court(self, points):
        """
        Given 4 points (top-left, top-right, bottom-right, bottom-left) in pixels,
        compute the homography matrix to map to a 20x10 meter padel court.
        """
        # Padel court is 20m x 10m
        dst_pts = np.array([
            [0, 0],
            [10, 0],
            [10, 20],
            [0, 20]
        ], dtype="float32")
        
        src_pts = np.array(points, dtype="float32")
        self.court_homography, _ = cv2.findHomography(src_pts, dst_pts)

    def get_real_world_coord(self, x, y):
        if self.court_homography is None:
            return None
        
        point = np.array([[[x, y]]], dtype="float32")
        real_coord = cv2.perspectiveTransform(point, self.court_homography)
        return real_coord[0][0]
