import numpy as np
from typing import Optional, List
from ultralytics import YOLO
from cv.detectors.base import BallDetector, PlayerDetector
from cv.detectors.device import get_device


class UnifiedYoloDetector:
    """Runs YOLO once per frame and caches results for both ball and player detectors."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        self._device = get_device()
        self.model = YOLO(model_path)
        self.model.to(self._device)
        self._cache = (None, None)  # (frame_id, results)

    def run(self, frame: np.ndarray, frame_id: int):
        if frame_id != self._cache[0]:
            results = self.model(frame, verbose=False)
            self._cache = (frame_id, results[0])
        return self._cache[1]

    @property
    def device(self) -> str:
        return self._device

    def warm_up(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)


class YoloBallDetector(BallDetector):
    CLS_SPORTS_BALL = 32

    def __init__(self, unified: UnifiedYoloDetector, conf_threshold: float = 0.3):
        self._unified = unified
        self._conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        result = self._unified.run(frame, frame_id)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        mask = (cls == self.CLS_SPORTS_BALL) & (conf >= self._conf_threshold)
        if not mask.any():
            return None

        filtered_conf = conf[mask]
        filtered_xyxy = xyxy[mask]
        best_idx = filtered_conf.argmax()
        return filtered_xyxy[best_idx].tolist()

    def warm_up(self) -> None:
        self._unified.warm_up()

    @property
    def device(self) -> str:
        return self._unified.device


class YoloPlayerDetector(PlayerDetector):
    CLS_PERSON = 0

    def __init__(self, unified: UnifiedYoloDetector,
                 conf_threshold: float = 0.6, max_detections: int = 4):
        self._unified = unified
        self._conf_threshold = conf_threshold
        self._max_detections = max_detections

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> np.ndarray:
        result = self._unified.run(frame, frame_id)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        mask = (cls == self.CLS_PERSON) & (conf >= self._conf_threshold)
        if not mask.any():
            return np.empty((0, 6))

        filtered_xyxy = xyxy[mask]
        filtered_cls = cls[mask]
        filtered_conf = conf[mask]

        sort_idx = filtered_conf.argsort()[::-1][:self._max_detections]
        out = np.column_stack([
            filtered_xyxy[sort_idx],
            filtered_conf[sort_idx],
            filtered_cls[sort_idx],
        ])
        return out

    def warm_up(self) -> None:
        self._unified.warm_up()

    @property
    def device(self) -> str:
        return self._unified.device
