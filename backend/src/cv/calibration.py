import numpy as np
import cv2

class CameraCalibration:
    def __init__(self, court_width=10, court_length=20):
        self.court_width = court_width
        self.court_length = court_length
        # Default Intrinsic Matrix (Assumes standard 1080p camera)
        self.K = np.array([
            [1000, 0, 960],
            [0, 1000, 540],
            [0, 0, 1]
        ], dtype=np.float32)

    def calculate_projection_matrix(self, pos_x, pos_y, height, tilt_deg, pan_deg):
        """
        Calculates the P matrix based on Virtual Camera settings.
        pos_x, pos_y: Camera position on the 20x10 court.
        height: Height in meters.
        tilt_deg: Downward angle (0 is horizontal, -90 is vertical).
        pan_deg: Horizontal rotation.
        """
        # 1. Rotation Matrices
        tilt_rad = np.radians(tilt_deg)
        pan_rad = np.radians(pan_deg)

        R_pan = np.array([
            [np.cos(pan_rad), -np.sin(pan_rad), 0],
            [np.sin(pan_rad), np.cos(pan_rad), 0],
            [0, 0, 1]
        ])

        R_tilt = np.array([
            [1, 0, 0],
            [0, np.cos(tilt_rad), -np.sin(tilt_rad)],
            [0, np.sin(tilt_rad), np.cos(tilt_rad)]
        ])

        R = R_tilt @ R_pan

        # 2. Translation Vector (Camera position in World)
        t = np.array([[-pos_x], [-pos_y], [-height]], dtype=np.float32)
        
        # 3. Combine into Extrinsic Matrix [R | t]
        Rt = np.hstack((R, t))
        
        # 4. Final Projection Matrix P = K * [R | t]
        P = self.K @ Rt
        return P

    def world_to_pixel(self, x, y, z, P):
        """Maps 3D Court (meters) to 2D Image (pixels)"""
        world_pt = np.array([x, y, z, 1.0], dtype=np.float32)
        pixel_pt = P @ world_pt
        pixel_pt /= pixel_pt[2] # Normalize by depth
        return int(pixel_pt[0]), int(pixel_pt[1])
