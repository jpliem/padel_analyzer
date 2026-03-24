import numpy as np
import pytest
from cv.player_tracker import PlayerTracker
from cv.court_calibration import CourtCalibration


@pytest.fixture
def calibrated_court(sample_court_corners_pixels):
    cal = CourtCalibration()
    cal.calibrate(sample_court_corners_pixels)
    return cal


@pytest.fixture
def mock_detections():
    """Simulated YOLO detections: Nx6 array [x1, y1, x2, y2, conf, cls]."""
    return np.array([
        [100, 200, 160, 400, 0.9, 0],
        [400, 200, 460, 400, 0.85, 0],
        [800, 500, 860, 700, 0.88, 0],
        [1100, 500, 1160, 700, 0.92, 0],
    ])


class TestPlayerDetection:
    def test_detect_players_returns_positions(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = tracker.update(mock_detections, frame_number=0)
        assert len(positions) > 0

    def test_player_positions_have_court_coords(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = tracker.update(mock_detections, frame_number=0)
        for pos in positions:
            assert "x" in pos and "y" in pos
            assert "track_id" in pos

    def test_empty_detections(self, calibrated_court):
        tracker = PlayerTracker(calibrated_court)
        positions = tracker.update(np.array([]).reshape(0, 6), frame_number=0)
        assert len(positions) == 0


class TestPlayerAssignment:
    def test_assign_player_id(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        tracker.assign_player(track_id=1, player_id="P1")
        assert tracker.get_player_id(track_id=1) == "P1"

    def test_unassigned_returns_none(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        assert tracker.get_player_id(track_id=999) is None

    def test_get_player_position(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        tracker.assign_player(track_id=1, player_id="P1")
        pos = tracker.get_player_position("P1")
        assert pos is not None
        assert "x" in pos and "y" in pos


class TestClosestPlayer:
    def test_find_closest_player(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        tracker.assign_player(track_id=1, player_id="P1")
        tracker.assign_player(track_id=2, player_id="P2")
        p1_pos = tracker.get_player_position("P1")
        closest = tracker.find_closest_player(p1_pos["x"], p1_pos["y"])
        assert closest == "P1"
