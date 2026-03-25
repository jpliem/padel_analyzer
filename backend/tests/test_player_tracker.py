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
def on_court_detections():
    """Detections that map to on-court positions with test calibration corners.
    Using pixels near center of the court area."""
    return np.array([
        [900, 300, 1000, 600, 0.9, 0],   # ~court center
        [500, 300, 600, 600, 0.85, 0],    # ~left side
        [900, 150, 1000, 300, 0.88, 0],   # ~far side
        [750, 150, 850, 300, 0.92, 0],    # ~far center
    ])


def _feed_frames(tracker, detections, n_frames=5):
    """Feed the same detections for N frames so ByteTrack establishes tracks."""
    positions = []
    for i in range(n_frames):
        positions = tracker.update(detections, frame_number=i)
    return positions


class TestPlayerDetection:
    def test_detect_players_returns_positions(self, calibrated_court, on_court_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = _feed_frames(tracker, on_court_detections)
        assert len(positions) > 0

    def test_player_positions_have_court_coords(self, calibrated_court, on_court_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = _feed_frames(tracker, on_court_detections)
        for pos in positions:
            assert "x" in pos and "y" in pos
            assert "track_id" in pos

    def test_empty_detections(self, calibrated_court):
        tracker = PlayerTracker(calibrated_court)
        positions = tracker.update(np.array([]).reshape(0, 6), frame_number=0)
        assert len(positions) == 0


class TestPlayerAssignment:
    def test_assign_player_id(self, calibrated_court, on_court_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = _feed_frames(tracker, on_court_detections)
        if positions:
            tid = positions[0]["track_id"]
            tracker.assign_player(track_id=tid, player_id="P1")
            assert tracker.get_player_id(track_id=tid) == "P1"

    def test_unassigned_returns_none(self, calibrated_court, on_court_detections):
        tracker = PlayerTracker(calibrated_court)
        _feed_frames(tracker, on_court_detections)
        assert tracker.get_player_id(track_id=999) is None

    def test_get_player_position(self, calibrated_court, on_court_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = _feed_frames(tracker, on_court_detections)
        if positions:
            tid = positions[0]["track_id"]
            tracker.assign_player(track_id=tid, player_id="P1")
            pos = tracker.get_player_position("P1")
            assert pos is not None
            assert "x" in pos and "y" in pos


class TestClosestPlayer:
    def test_find_closest_player(self, calibrated_court, on_court_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = _feed_frames(tracker, on_court_detections)
        if len(positions) >= 2:
            tracker.assign_player(track_id=positions[0]["track_id"], player_id="P1")
            tracker.assign_player(track_id=positions[1]["track_id"], player_id="P2")
            p1_pos = tracker.get_player_position("P1")
            if p1_pos:
                closest = tracker.find_closest_player(p1_pos["x"], p1_pos["y"])
                assert closest == "P1"
