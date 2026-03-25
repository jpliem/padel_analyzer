"""Tests for CameraNode — per-camera processing wrapper."""

import math
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from cv.camera_node import CameraNode, CameraHealth
from cv.camera_model import CameraModel
from models.types import CameraObservation, BallPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_calibrated_node(camera_id: str = "cam0") -> CameraNode:
    """Return a CameraNode that has been calibrated with 12 ground keypoints."""
    node = CameraNode(camera_id, label="Test Camera")
    # Realistic pixel coords for a behind-baseline camera looking at a 10×20 m court
    keypoints_2d = [
        [320, 700],   # k1  near-left baseline
        [1600, 700],  # k2  near-right baseline
        [480, 540],   # k3  near-left service
        [960, 540],   # k4  near-center service
        [1440, 540],  # k5  near-right service
        [600, 450],   # k6  net-left (ground)
        [1320, 450],  # k7  net-right (ground)
        [660, 380],   # k8  far-left service
        [960, 380],   # k9  far-center service
        [1260, 380],  # k10 far-right service
        [720, 280],   # k11 far-left baseline
        [1200, 280],  # k12 far-right baseline
    ]
    node.calibrate(keypoints_2d)
    return node


def make_frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestCameraNodeCreation:
    def test_defaults(self):
        node = CameraNode("cam1")
        assert node.camera_id == "cam1"
        assert node.label == ""
        assert node.detector_type == "yolo"
        assert node.health == CameraHealth.UNCALIBRATED
        assert isinstance(node.camera_model, CameraModel)

    def test_custom_label_and_detector_type(self):
        node = CameraNode("cam2", label="Side Camera", detector_type="yolo")
        assert node.label == "Side Camera"
        assert node.detector_type == "yolo"

    def test_initial_reprojection_error_is_inf(self):
        node = CameraNode("cam0")
        assert math.isinf(node._reprojection_error)

    def test_detectors_start_as_none(self):
        node = CameraNode("cam0")
        assert node._ball_detector is None
        assert node._player_detector is None

    def test_last_frame_time_starts_at_zero(self):
        node = CameraNode("cam0")
        assert node._last_frame_time == 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_calibrate_sets_health_to_active(self):
        node = make_calibrated_node()
        assert node.health == CameraHealth.ACTIVE

    def test_calibrate_sets_calibrated_flag(self):
        node = make_calibrated_node()
        assert node._calibrated is True

    def test_camera_model_accessible_after_calibration(self):
        node = make_calibrated_node()
        assert node.camera_model is not None
        # homography should be populated (12 ground points → findHomography)
        assert node.camera_model.homography is not None


# ---------------------------------------------------------------------------
# process_frame — ball found
# ---------------------------------------------------------------------------

class TestProcessFrameBallFound:
    def setup_method(self):
        self.node = make_calibrated_node()

        # Mock ball detector: returns a bbox
        self.mock_ball = MagicMock()
        self.mock_ball.detect.return_value = [600.0, 340.0, 620.0, 360.0]  # [x1,y1,x2,y2]

        # Mock player detector: returns empty array (no players)
        self.mock_player = MagicMock()
        self.mock_player.detect.return_value = np.empty((0, 6))

        self.node._ball_detector = self.mock_ball
        self.node._player_detector = self.mock_player

    def test_returns_camera_observation(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert isinstance(obs, CameraObservation)

    def test_camera_id_in_observation(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.camera_id == "cam0"

    def test_ball_pixel_is_bbox_center(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        # center of [600, 340, 620, 360] → (610, 350)
        assert obs.ball_pixel == pytest.approx((610.0, 350.0))

    def test_ball_bbox_stored(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.ball_bbox == [600.0, 340.0, 620.0, 360.0]

    def test_ball_court_is_ball_position(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.ball_court is not None
        assert isinstance(obs.ball_court, BallPosition)

    def test_ball_court_within_court_bounds(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.ball_court is not None
        # Court is 10 m wide × 20 m long; allow generous margin because the
        # mock keypoints produce a rough calibration that may project to just
        # outside the strict boundary.
        assert -5.0 <= obs.ball_court.x <= 15.0
        assert -5.0 <= obs.ball_court.y <= 25.0

    def test_confidence_set(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.confidence == 1.0

    def test_timestamp_and_frame_number_stored(self):
        obs = self.node.process_frame(make_frame(), frame_number=42, timestamp=3.5)
        assert obs.timestamp == 3.5
        assert obs.frame_number == 42

    def test_last_frame_time_updated(self):
        self.node.process_frame(make_frame(), frame_number=1, timestamp=7.25)
        assert self.node._last_frame_time == pytest.approx(7.25)


# ---------------------------------------------------------------------------
# process_frame — ball not found
# ---------------------------------------------------------------------------

class TestProcessFrameBallNotFound:
    def setup_method(self):
        self.node = make_calibrated_node()

        self.mock_ball = MagicMock()
        self.mock_ball.detect.return_value = None  # no ball

        self.mock_player = MagicMock()
        self.mock_player.detect.return_value = np.empty((0, 6))

        self.node._ball_detector = self.mock_ball
        self.node._player_detector = self.mock_player

    def test_ball_pixel_is_none(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.ball_pixel is None

    def test_ball_bbox_is_none(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.ball_bbox is None

    def test_ball_court_is_none(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.ball_court is None

    def test_confidence_is_zero(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.confidence == 0.0


# ---------------------------------------------------------------------------
# process_frame — player detection
# ---------------------------------------------------------------------------

class TestProcessFramePlayers:
    def setup_method(self):
        self.node = make_calibrated_node()

        self.mock_ball = MagicMock()
        self.mock_ball.detect.return_value = None

        # Two player detections: [x1, y1, x2, y2, conf, cls]
        self.mock_player = MagicMock()
        self.mock_player.detect.return_value = np.array([
            [100.0, 400.0, 200.0, 700.0, 0.9, 0.0],
            [800.0, 420.0, 900.0, 710.0, 0.8, 0.0],
        ])

        self.node._ball_detector = self.mock_ball
        self.node._player_detector = self.mock_player

    def test_two_players_detected(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert len(obs.player_detections) == 2

    def test_player_entry_has_bbox_pixel(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        p = obs.player_detections[0]
        assert "bbox_pixel" in p
        assert p["bbox_pixel"] == [100.0, 400.0, 200.0, 700.0]

    def test_player_entry_has_foot_pixel(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        p = obs.player_detections[0]
        # foot_pixel = bottom-centre of bbox → x=(100+200)/2=150, y=700
        assert "foot_pixel" in p
        assert p["foot_pixel"] == pytest.approx((150.0, 700.0))

    def test_player_entry_has_confidence(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        assert obs.player_detections[0]["confidence"] == pytest.approx(0.9)

    def test_player_entry_has_court_coords(self):
        obs = self.node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        p = obs.player_detections[0]
        assert "court" in p
        assert "x" in p["court"]
        assert "y" in p["court"]


# ---------------------------------------------------------------------------
# process_frame — uncalibrated camera
# ---------------------------------------------------------------------------

class TestProcessFrameUncalibrated:
    def test_returns_empty_observation_when_not_calibrated(self):
        node = CameraNode("cam_raw")  # not calibrated

        mock_ball = MagicMock()
        mock_ball.detect.return_value = [100.0, 100.0, 120.0, 120.0]
        node._ball_detector = mock_ball

        obs = node.process_frame(make_frame(), frame_number=1, timestamp=1.0)
        # Should return early without projecting anything
        assert obs.ball_pixel is None
        assert obs.ball_court is None


# ---------------------------------------------------------------------------
# Health management
# ---------------------------------------------------------------------------

class TestHealthManagement:
    def test_uncalibrated_stays_uncalibrated_on_update(self):
        node = CameraNode("cam0")
        node.update_health(current_time=100.0)
        assert node.health == CameraHealth.UNCALIBRATED

    def test_active_when_gap_is_small(self):
        node = make_calibrated_node()
        node._last_frame_time = 10.0
        node.update_health(current_time=11.0)  # gap = 1 s < DEGRADED_TIMEOUT
        assert node.health == CameraHealth.ACTIVE

    def test_degraded_after_timeout(self):
        node = make_calibrated_node()
        node._last_frame_time = 10.0
        node.update_health(current_time=13.0)  # gap = 3 s > DEGRADED_TIMEOUT
        assert node.health == CameraHealth.DEGRADED

    def test_degraded_exactly_at_boundary(self):
        node = make_calibrated_node()
        node._last_frame_time = 0.0
        # gap = DEGRADED_TIMEOUT + epsilon → DEGRADED
        node.update_health(current_time=CameraNode.DEGRADED_TIMEOUT + 0.001)
        assert node.health == CameraHealth.DEGRADED

    def test_disconnected_after_long_gap(self):
        node = make_calibrated_node()
        node._last_frame_time = 0.0
        node.update_health(current_time=15.0)  # gap = 15 s > DISCONNECT_TIMEOUT
        assert node.health == CameraHealth.DISCONNECTED

    def test_disconnected_exactly_at_boundary(self):
        node = make_calibrated_node()
        node._last_frame_time = 0.0
        node.update_health(current_time=CameraNode.DISCONNECT_TIMEOUT + 0.001)
        assert node.health == CameraHealth.DISCONNECTED

    def test_health_recovers_to_active_after_recent_frame(self):
        node = make_calibrated_node()
        node._last_frame_time = 0.0
        node.update_health(current_time=15.0)
        assert node.health == CameraHealth.DISCONNECTED

        # Simulate a new frame arriving
        node._last_frame_time = 15.0
        node.update_health(current_time=15.5)  # gap = 0.5 s → ACTIVE
        assert node.health == CameraHealth.ACTIVE


# ---------------------------------------------------------------------------
# Quality weight
# ---------------------------------------------------------------------------

class TestQualityWeight:
    def test_uncalibrated_returns_zero(self):
        node = CameraNode("cam0")
        assert node.quality_weight() == 0.0

    def test_disconnected_returns_zero(self):
        node = make_calibrated_node()
        node._health = CameraHealth.DISCONNECTED
        assert node.quality_weight() == 0.0

    def test_high_reprojection_error_returns_zero(self):
        node = make_calibrated_node()
        node._reprojection_error = 16.0
        assert node.quality_weight() == 0.0

    def test_medium_reprojection_error_returns_half(self):
        node = make_calibrated_node()
        node._reprojection_error = 10.0
        assert node.quality_weight() == 0.5

    def test_low_reprojection_error_returns_one(self):
        node = make_calibrated_node()
        node._reprojection_error = 3.0
        assert node.quality_weight() == 1.0

    def test_inf_reprojection_error_returns_zero(self):
        # Default state after construction (not yet calibrated)
        node = make_calibrated_node()
        # _reprojection_error starts as inf
        node._reprojection_error = math.inf
        # health is ACTIVE after calibration, but error is inf > 15 → 0.0
        assert node.quality_weight() == 0.0

    def test_exactly_at_15px_boundary_returns_zero(self):
        node = make_calibrated_node()
        node._reprojection_error = 15.001
        assert node.quality_weight() == 0.0

    def test_exactly_at_5px_boundary_returns_half(self):
        node = make_calibrated_node()
        node._reprojection_error = 5.001
        assert node.quality_weight() == 0.5

    def test_degraded_health_with_good_error_returns_one(self):
        """DEGRADED camera should still contribute if reprojection error is good."""
        node = make_calibrated_node()
        node._health = CameraHealth.DEGRADED
        node._reprojection_error = 2.0
        assert node.quality_weight() == 1.0


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_health_property(self):
        node = CameraNode("cam0")
        assert node.health is node._health

    def test_camera_model_property(self):
        node = CameraNode("cam0")
        assert node.camera_model is node._camera_model
