import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List

COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
NET_Y = 10.0
SERVICE_NEAR_Y = 6.95
SERVICE_FAR_Y = 13.05
SERVICE_CENTER_X = 5.0

# 12 keypoints in court coordinates (meters):
#
# k11(0,20)-------------------k12(10,20)
# |                              |
# k8(0,13.05)---k9(5,13.05)---k10(10,13.05)
# |             |                |
# |             |                |
# k6(0,10)-----+---------- ---k7(10,10)    ← NET
# |             |                |
# |             |                |
# k3(0,6.95)---k4(5,6.95)-----k5(10,6.95)
# |                              |
# k1(0,0)----------------------k2(10,0)

KEYPOINT_COURT_COORDS_12 = [
    (0, 0),                           # k1  near-left baseline
    (COURT_WIDTH, 0),                  # k2  near-right baseline
    (0, SERVICE_NEAR_Y),              # k3  near-left service
    (SERVICE_CENTER_X, SERVICE_NEAR_Y), # k4  near-center service
    (COURT_WIDTH, SERVICE_NEAR_Y),     # k5  near-right service
    (0, NET_Y),                        # k6  net-left
    (COURT_WIDTH, NET_Y),              # k7  net-right
    (0, SERVICE_FAR_Y),               # k8  far-left service
    (SERVICE_CENTER_X, SERVICE_FAR_Y), # k9  far-center service
    (COURT_WIDTH, SERVICE_FAR_Y),      # k10 far-right service
    (0, COURT_LENGTH),                 # k11 far-left baseline
    (COURT_WIDTH, COURT_LENGTH),       # k12 far-right baseline
]


class CourtCalibration:
    NET_Y = NET_Y

    def __init__(self):
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.corners_pixels: Optional[np.ndarray] = None

    def calibrate(self, corners_pixels: np.ndarray,
                  net_pixels: Optional[np.ndarray] = None) -> None:
        """Calibrate with 4 corners + optional net points (legacy 4-6 point mode)."""
        if len(corners_pixels) != 4:
            raise ValueError("Exactly 4 corner points required")
        self.corners_pixels = corners_pixels.astype(np.float32)

        src_points = list(corners_pixels.astype(np.float32))
        dst_points = [
            [0, 0], [COURT_WIDTH, 0],
            [COURT_WIDTH, COURT_LENGTH], [0, COURT_LENGTH],
        ]

        if net_pixels is not None and len(net_pixels) == 2:
            src_points.append(net_pixels[0].astype(np.float32))
            src_points.append(net_pixels[1].astype(np.float32))
            dst_points.append([0, NET_Y])
            dst_points.append([COURT_WIDTH, NET_Y])

        src = np.array(src_points, dtype=np.float32)
        dst = np.array(dst_points, dtype=np.float32)
        self.homography, _ = cv2.findHomography(src, dst)
        self.inverse_homography, _ = cv2.findHomography(dst, src)

    def calibrate_keypoints(self, keypoints_pixels: List[List[float]]) -> None:
        """Calibrate with 12 court keypoints for maximum accuracy.

        Args:
            keypoints_pixels: List of [x, y] pixel coordinates for each of the 12 keypoints.
                Order: k1-k12 as defined in KEYPOINT_COURT_COORDS_12.
                Can also accept fewer points (minimum 4) — will use only provided points.
        """
        n = len(keypoints_pixels)
        if n < 4:
            raise ValueError(f"Need at least 4 keypoints, got {n}")

        src = np.array(keypoints_pixels[:n], dtype=np.float32)
        dst = np.array(KEYPOINT_COURT_COORDS_12[:n], dtype=np.float32)

        self.homography, _ = cv2.findHomography(src, dst)
        self.inverse_homography, _ = cv2.findHomography(dst, src)
        # Store corners (k1, k2, k12, k11) for polygon filtering
        if n >= 12:
            self.corners_pixels = np.array([
                keypoints_pixels[0], keypoints_pixels[1],
                keypoints_pixels[11], keypoints_pixels[10],
            ], dtype=np.float32)
        else:
            self.corners_pixels = src[:4]

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
            "corners_pixels": self.corners_pixels.tolist() if self.corners_pixels is not None else None,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CourtCalibration":
        cal = cls()
        cal.homography = np.array(data["homography"], dtype=np.float64)
        cal.inverse_homography = np.array(data["inverse_homography"], dtype=np.float64)
        if data.get("corners_pixels"):
            cal.corners_pixels = np.array(data["corners_pixels"], dtype=np.float32)
        return cal
