import cv2
import numpy as np
from typing import Tuple, Optional, Dict

COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
NET_Y = 10.0
SERVICE_NEAR_Y = 6.95
SERVICE_FAR_Y = 13.05
SERVICE_CENTER_X = 5.0


class CourtCalibration:
    NET_Y = NET_Y

    def __init__(self):
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.corners_pixels: Optional[np.ndarray] = None

    def calibrate(self, corners_pixels: np.ndarray) -> None:
        if len(corners_pixels) != 4:
            raise ValueError("Exactly 4 corner points required")
        self.corners_pixels = corners_pixels.astype(np.float32)
        court_corners = np.array([
            [0, 0], [COURT_WIDTH, 0],
            [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH],
        ], dtype=np.float32)
        self.homography, _ = cv2.findHomography(corners_pixels, court_corners)
        self.inverse_homography, _ = cv2.findHomography(court_corners, corners_pixels)

    def _check_calibrated(self):
        if self.homography is None:
            raise RuntimeError("Court not calibrated — call calibrate() first")

    def pixel_to_court(self, px: float, py: float) -> Tuple[float, float]:
        self._check_calibrated()
        point = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.homography)
        return float(result[0][0][0]), float(result[0][0][1])

    def court_to_pixel(self, cx: float, cy: float) -> Tuple[float, float]:
        self._check_calibrated()
        point = np.array([[[cx, cy]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.inverse_homography)
        return float(result[0][0][0]), float(result[0][0][1])

    def is_in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x <= COURT_WIDTH and 0 <= y <= COURT_LENGTH

    def get_court_side(self, x: float, y: float) -> str:
        return "near" if y < NET_Y else "far"

    def is_in_service_box(self, x: float, y: float, box: str) -> bool:
        boxes = {
            "near_left": (0, SERVICE_CENTER_X, SERVICE_NEAR_Y, NET_Y),
            "near_right": (SERVICE_CENTER_X, COURT_WIDTH, SERVICE_NEAR_Y, NET_Y),
            "far_left": (0, SERVICE_CENTER_X, NET_Y, SERVICE_FAR_Y),
            "far_right": (SERVICE_CENTER_X, COURT_WIDTH, NET_Y, SERVICE_FAR_Y),
        }
        if box not in boxes:
            return False
        x1, x2, y1, y2 = boxes[box]
        return x1 <= x <= x2 and y1 <= y <= y2

    def to_dict(self) -> Dict:
        self._check_calibrated()
        return {
            "homography": self.homography.tolist(),
            "inverse_homography": self.inverse_homography.tolist(),
            "corners_pixels": self.corners_pixels.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CourtCalibration":
        cal = cls()
        cal.homography = np.array(data["homography"], dtype=np.float64)
        cal.inverse_homography = np.array(data["inverse_homography"], dtype=np.float64)
        cal.corners_pixels = np.array(data["corners_pixels"], dtype=np.float32)
        return cal
