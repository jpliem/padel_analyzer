"""Automatic court keypoint detection using YOLO pose model.

The model detects 12 court keypoints per frame. Output order from the model
needs to be mapped to our standard court coordinate order (KEYPOINT_COURT_COORDS_12).

Model output order (by pixel position analysis):
  0,1: far baseline (top of frame)
  2,3: near baseline (bottom of frame) — NOTE: right then left
  4,5: far service line (left, right)
  6: far service center
  7,8: net line (left, right)
  9,10: near service line (left, right)
  11: near service center

Our standard order (KEYPOINT_COURT_COORDS_12):
  0: k1  near-left baseline
  1: k2  near-right baseline
  2: k3  near-left service
  3: k4  near-center service
  4: k5  near-right service
  5: k6  net-left
  6: k7  net-right
  7: k8  far-left service
  8: k9  far-center service
  9: k10 far-right service
  10: k11 far-left baseline
  11: k12 far-right baseline
"""

import numpy as np
from typing import Optional, List, Tuple
from ultralytics import YOLO
from cv.detectors.device import get_device

# Mapping: model output index → our standard keypoint index
# This maps the YOLO model's 12 keypoint outputs to our k1-k12 order
MODEL_TO_COURT_MAP = {
    0: 10,  # model[0] (far-left baseline) → k11
    1: 11,  # model[1] (far-right baseline) → k12
    2: 1,   # model[2] (near-right baseline) → k2
    3: 0,   # model[3] (near-left baseline) → k1
    4: 7,   # model[4] (far-left service) → k8
    5: 9,   # model[5] (far-right service) → k10
    6: 8,   # model[6] (far-center service) → k9
    7: 5,   # model[7] (net-left) → k6
    8: 6,   # model[8] (net-right) → k7
    9: 2,   # model[9] (near-left service) → k3
    10: 4,  # model[10] (near-right service) → k5
    11: 3,  # model[11] (near-center service) → k4
}


class CourtDetector:
    """Automatic court keypoint detection using YOLO pose model."""

    def __init__(self, model_path: str = "models/court_keypoints.pt",
                 conf_threshold: float = 0.3):
        self._device = get_device()
        self._model = YOLO(model_path)
        self._model.to(self._device)
        self._conf_threshold = conf_threshold
        self._last_keypoints: Optional[np.ndarray] = None

    def detect(self, frame: np.ndarray) -> Optional[List[List[float]]]:
        """Detect 12 court keypoints in a frame.

        Returns:
            List of 12 [x, y] pixel coordinates in standard k1-k12 order,
            or None if detection failed.
        """
        results = self._model(frame, verbose=False)
        if not results or len(results) == 0:
            return self._last_keypoints_as_list()

        r = results[0]
        if r.keypoints is None or r.keypoints.xy is None:
            return self._last_keypoints_as_list()

        if r.keypoints.xy.shape[0] == 0:
            return self._last_keypoints_as_list()
        kps = r.keypoints.xy[0].cpu().numpy()  # (12, 2)
        if len(kps) != 12:
            return self._last_keypoints_as_list()

        # Check confidence
        if r.keypoints.conf is not None:
            conf = r.keypoints.conf[0].cpu().numpy()
            if conf.mean() < self._conf_threshold:
                return self._last_keypoints_as_list()

        # Filter out zero/invalid keypoints
        valid = np.all(kps > 0, axis=1)
        if valid.sum() < 8:  # need at least 8 valid keypoints
            return self._last_keypoints_as_list()

        # Remap from model order to our standard k1-k12 order
        reordered = np.zeros((12, 2), dtype=np.float64)
        for model_idx, court_idx in MODEL_TO_COURT_MAP.items():
            if valid[model_idx]:
                reordered[court_idx] = kps[model_idx]
            elif self._last_keypoints is not None:
                reordered[court_idx] = self._last_keypoints[court_idx]

        self._last_keypoints = reordered
        return reordered.tolist()

    def _last_keypoints_as_list(self) -> Optional[List[List[float]]]:
        if self._last_keypoints is not None:
            return self._last_keypoints.tolist()
        return None

    def warm_up(self, frame: np.ndarray) -> None:
        """Run first detection to warm up the model."""
        self.detect(frame)
