"""Optional COCO pose adapter and racket-hand proximity evidence."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


LEFT_WRIST = 9
RIGHT_WRIST = 10


@dataclass(frozen=True)
class PlayerPose:
    bbox: Tuple[float, float, float, float]
    keypoints: Tuple[Tuple[float, float, float], ...]
    confidence: float


def wrist_proximity_confidence(ball_pixel: Tuple[float, float],
                                keypoints: Sequence[Sequence[float]],
                                radius_px: float = 80.0) -> float:
    """Return image-space hand proximity; this is evidence, not a hit label."""
    distances = []
    for index in (LEFT_WRIST, RIGHT_WRIST):
        if index >= len(keypoints):
            continue
        point = keypoints[index]
        if len(point) < 2 or (len(point) >= 3 and point[2] < 0.2):
            continue
        distances.append(float(np.linalg.norm(
            np.asarray(point[:2], dtype=float) - np.asarray(ball_pixel, dtype=float))))
    if not distances:
        return 0.0
    return max(0.0, 1.0 - min(distances) / max(radius_px, 1e-6))


class PlayerPoseDetector:
    """Ultralytics pose backend loaded only when explicitly configured."""

    def __init__(self, model_path: str, confidence: float = 0.35):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> List[PlayerPose]:
        result = self.model(frame, verbose=False, conf=self.confidence)[0]
        if result.keypoints is None or result.boxes is None:
            return []
        xy = result.keypoints.xy.cpu().numpy()
        kp_conf = (result.keypoints.conf.cpu().numpy()
                   if result.keypoints.conf is not None else np.ones(xy.shape[:2]))
        boxes = result.boxes.xyxy.cpu().numpy()
        box_conf = result.boxes.conf.cpu().numpy()
        poses = []
        for bbox, points, point_conf, confidence in zip(boxes, xy, kp_conf, box_conf):
            keypoints = tuple((float(p[0]), float(p[1]), float(c))
                              for p, c in zip(points, point_conf))
            poses.append(PlayerPose(tuple(float(v) for v in bbox), keypoints,
                                    float(confidence)))
        return poses

