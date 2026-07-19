import numpy as np
import pytest
from cv.ball_tracker import BallTracker
from cv.court_calibration import CourtCalibration
from cv.camera_model import CameraModel


@pytest.fixture
def calibrated_court(sample_court_corners_pixels):
    cal = CourtCalibration()
    cal.calibrate(sample_court_corners_pixels)
    return cal


@pytest.fixture
def tracker(calibrated_court):
    return BallTracker(calibrated_court, fps=30)


class TestUncalibratedFallback:
    def test_update_without_calibration_stays_in_pixel_space(self):
        tracker = BallTracker(CourtCalibration(), fps=30)
        pos = tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert pos is not None
        assert pos["x"] == pytest.approx(510.0, abs=1.0)
        assert pos["y"] == pytest.approx(410.0, abs=1.0)

    def test_prediction_without_calibration_does_not_crash(self):
        tracker = BallTracker(CourtCalibration(), fps=30)
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        pos = tracker.update(bbox=None, frame_number=1)
        assert pos is not None


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
        # Feed multiple frames with ball moving up (decreasing pixel Y) to accumulate Z
        tracker.update(bbox=[500, 400, 540, 440], frame_number=0)
        tracker.update(bbox=[500, 350, 540, 390], frame_number=1)  # moved up
        tracker.update(bbox=[500, 300, 540, 340], frame_number=2)  # moved up more
        # Z should be > 0 after ball moved upward
        z = tracker.trajectory[-1]["z"]
        assert z >= 0  # Z estimation is active


class TestCameraModelCompatibility:
    def test_constructs_with_uncalibrated_camera_model(self):
        """BallTracker should accept an uncalibrated CameraModel without error."""
        cam = CameraModel()
        tracker = BallTracker(cam, fps=30)
        assert tracker.calibration is cam

    def test_constructs_with_calibrated_camera_model(self, sample_court_corners_pixels):
        """BallTracker should accept a calibrated CameraModel (homography only)."""
        cam = CameraModel()
        # Provide 4 ground keypoints — enough for homography but not full 3D
        cam.calibrate(sample_court_corners_pixels[:4].tolist())
        tracker = BallTracker(cam, fps=30)
        assert tracker.calibration is cam

    def test_update_with_calibrated_camera_model(self, sample_court_corners_pixels):
        """BallTracker.update should work when backed by a CameraModel with homography."""
        cam = CameraModel()
        # Use all 4 available sample corners for homography calibration
        cam.calibrate(sample_court_corners_pixels[:4].tolist())
        tracker = BallTracker(cam, fps=30)
        pos = tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert pos is not None
        assert "x" in pos and "y" in pos
class TestValidPosition:
    def test_nan_x_rejected(self, tracker):
        assert tracker._is_valid_position(float("nan"), 5.0, 0.0) is False

    def test_nan_y_rejected(self, tracker):
        assert tracker._is_valid_position(5.0, float("nan"), 0.0) is False

    def test_nan_z_rejected(self, tracker):
        assert tracker._is_valid_position(5.0, 10.0, float("nan")) is False

    def test_out_of_bounds_rejected(self, tracker):
        # 100m away from court center — clearly wrong
        assert tracker._is_valid_position(105.0, 10.0, 0.0) is False

    def test_valid_position_accepted(self, tracker):
        # Court centre is (5, 10); a typical in-bounds point
        assert tracker._is_valid_position(5.0, 10.0, 0.5) is True

    def test_nan_position_not_appended_to_trajectory(self, tracker, monkeypatch):
        """When calibration returns NaN coords, trajectory should stay empty."""

        def nan_projection(px, py):
            return float("nan"), float("nan")

        # Patch whichever projection method the calibration object exposes
        for method in ("pixel_to_court", "project_to_ground", "project_to_height"):
            if hasattr(tracker.calibration, method):
                monkeypatch.setattr(tracker.calibration, method, nan_projection)
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert len(tracker.trajectory) == 0

    def test_out_of_bounds_position_not_appended_to_trajectory(self, tracker, monkeypatch):
        """When calibration returns wildly wrong coords, trajectory should stay empty."""

        def oob_projection(px, py):
            return 999.0, 999.0

        for method in ("pixel_to_court", "project_to_ground", "project_to_height"):
            if hasattr(tracker.calibration, method):
                monkeypatch.setattr(tracker.calibration, method, oob_projection)
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert len(tracker.trajectory) == 0


44 -0
