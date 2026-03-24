import numpy as np
import pytest
from cv.ball_tracker import BallTracker
from cv.court_calibration import CourtCalibration


@pytest.fixture
def calibrated_court(sample_court_corners_pixels):
    cal = CourtCalibration()
    cal.calibrate(sample_court_corners_pixels)
    return cal


@pytest.fixture
def tracker(calibrated_court):
    return BallTracker(calibrated_court, fps=30)


class TestBallDetection:
    def test_update_with_detection(self, tracker):
        pos = tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert pos is not None
        assert "x" in pos and "y" in pos

    def test_update_without_detection_returns_prediction(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        tracker.update(bbox=[510, 405, 530, 425], frame_number=1)
        tracker.update(bbox=[520, 410, 540, 430], frame_number=2)
        pos = tracker.update(bbox=None, frame_number=3)
        assert pos is not None

    def test_no_prediction_without_initial_detection(self, tracker):
        pos = tracker.update(bbox=None, frame_number=0)
        assert pos is None

    def test_lost_after_many_misses(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        for i in range(1, 62):
            tracker.update(bbox=None, frame_number=i)
        assert tracker.is_lost is True


class TestTrajectory:
    def test_trajectory_stored(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        tracker.update(bbox=[510, 410, 530, 430], frame_number=1)
        assert len(tracker.trajectory) == 2

    def test_trajectory_has_court_coords(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        pos = tracker.trajectory[0]
        assert "x" in pos and "y" in pos and "timestamp" in pos


class TestSpeedEstimation:
    def test_speed_computed(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        tracker.update(bbox=[600, 400, 620, 420], frame_number=1)
        assert tracker.trajectory[-1]["speed"] >= 0

    def test_speed_zero_on_first_frame(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert tracker.trajectory[0]["speed"] == 0.0


class TestZEstimation:
    def test_z_estimated_from_bbox_size(self, tracker):
        tracker.update(bbox=[500, 400, 540, 440], frame_number=0)  # 40x40
        z1 = tracker.trajectory[0]["z"]
        tracker2 = BallTracker(tracker.calibration, fps=30)
        tracker2.update(bbox=[500, 400, 560, 460], frame_number=0)  # 60x60
        z2 = tracker2.trajectory[0]["z"]
        assert z2 > z1
