"""CameraNode — Per-camera processing wrapper.

Each physical camera in the multi-camera setup is represented by one CameraNode.
It owns a CameraModel (calibration), ball/player detectors, and health state.
process_frame() returns a CameraObservation for the WorldFusion layer.
"""

import math
from enum import Enum
from typing import Optional, List

import numpy as np

from cv.camera_model import CameraModel
from models.types import BallPosition, CameraObservation


class CameraHealth(Enum):
    UNCALIBRATED = "uncalibrated"
    ACTIVE = "active"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class CameraNode:
    DEGRADED_TIMEOUT = 2.0    # seconds without a frame → DEGRADED
    DISCONNECT_TIMEOUT = 10.0  # seconds without a frame → DISCONNECTED

    def __init__(self, camera_id: str, label: str = "", detector_type: str = "yolo"):
        self.camera_id = camera_id
        self.label = label
        self.detector_type = detector_type

        self._camera_model = CameraModel()
        self._calibrated = False
        self._health = CameraHealth.UNCALIBRATED

        self._ball_detector = None
        self._player_detector = None

        self._last_frame_time: float = 0.0
        self._reprojection_error: float = math.inf

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        keypoints_2d,
        net_top_2d=None,
        image_width: int = 1280,
        image_height: int = 720,
    ) -> None:
        """Calibrate the camera model from 2D keypoint observations."""
        self._camera_model.calibrate(
            keypoints_2d,
            net_top_2d=net_top_2d,
            image_width=image_width,
            image_height=image_height,
        )
        self._calibrated = True
        self._health = CameraHealth.ACTIVE

    # ------------------------------------------------------------------
    # Detector initialisation (skipped in tests — requires model files)
    # ------------------------------------------------------------------

    def init_detectors(self) -> None:
        """Initialise ball and player detectors (requires YOLO model files)."""
        if self.detector_type == "yolo":
            from cv.detectors.yolo import (
                UnifiedYoloDetector,
                YoloBallDetector,
                YoloPlayerDetector,
            )
            unified = UnifiedYoloDetector()
            self._ball_detector = YoloBallDetector(unified)
            self._player_detector = YoloPlayerDetector(unified)
        else:
            raise ValueError(f"Unknown detector_type: {self.detector_type!r}")

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame, frame_number: int, timestamp: float) -> CameraObservation:
        """Run detection on one frame and return a CameraObservation.

        Requires detectors to be initialised (via init_detectors) and the
        camera to be calibrated, otherwise returns an empty observation.
        """
        self._last_frame_time = timestamp

        obs = CameraObservation(
            camera_id=self.camera_id,
            timestamp=timestamp,
            frame_number=frame_number,
        )

        if not self._calibrated:
            return obs

        # --- Ball detection ---
        if self._ball_detector is not None:
            bbox = self._ball_detector.detect(frame, frame_id=frame_number)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cx_px = (x1 + x2) / 2.0
                cy_px = (y1 + y2) / 2.0
                obs.ball_pixel = (cx_px, cy_px)
                obs.ball_bbox = list(bbox)

                try:
                    court_x, court_y = self._camera_model.project_to_ground(cx_px, cy_px)
                    obs.ball_court = BallPosition(
                        x=court_x,
                        y=court_y,
                        z=0.0,
                        timestamp=timestamp,
                    )
                    obs.confidence = 1.0
                except Exception:
                    pass

        # --- Player detection ---
        if self._player_detector is not None:
            detections = self._player_detector.detect(frame, frame_id=frame_number)
            player_list: List[dict] = []
            for row in detections:
                x1, y1, x2, y2, conf, cls = row
                foot_px = ((x1 + x2) / 2.0, y2)  # bottom-centre of bbox

                player_entry: dict = {
                    "bbox_pixel": [x1, y1, x2, y2],
                    "foot_pixel": foot_px,
                    "confidence": float(conf),
                }

                try:
                    cx, cy = self._camera_model.project_to_ground(foot_px[0], foot_px[1])
                    player_entry["court"] = {"x": cx, "y": cy}
                except Exception:
                    pass

                player_list.append(player_entry)
            obs.player_detections = player_list

        return obs

    # ------------------------------------------------------------------
    # Health management
    # ------------------------------------------------------------------

    def update_health(self, current_time: float) -> None:
        """Update health state based on time since last frame."""
        if not self._calibrated:
            self._health = CameraHealth.UNCALIBRATED
            return

        gap = current_time - self._last_frame_time
        if gap > self.DISCONNECT_TIMEOUT:
            self._health = CameraHealth.DISCONNECTED
        elif gap > self.DEGRADED_TIMEOUT:
            self._health = CameraHealth.DEGRADED
        else:
            self._health = CameraHealth.ACTIVE

    # ------------------------------------------------------------------
    # Quality weight
    # ------------------------------------------------------------------

    def quality_weight(self) -> float:
        """Return a [0, 1] weight for fusion, based on health + reprojection error."""
        if self._health in (CameraHealth.DISCONNECTED, CameraHealth.UNCALIBRATED):
            return 0.0
        if self._reprojection_error > 15.0:
            return 0.0
        if self._reprojection_error > 5.0:
            return 0.5
        return 1.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def health(self) -> CameraHealth:
        return self._health

    @property
    def camera_model(self) -> CameraModel:
        return self._camera_model
