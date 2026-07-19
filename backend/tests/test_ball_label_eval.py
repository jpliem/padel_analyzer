import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.eval_ball_labels import compute_metrics, match_prediction, tracknet_prediction


class _FakeCapture:
    def __init__(self):
        self.position = 0

    def set(self, _property, value):
        self.position = int(value)

    def read(self):
        frame = self.position
        self.position += 1
        return True, frame


class _FakeTemporalDetector:
    N_INPUT_FRAMES = 3

    def __init__(self):
        self.frames = []
        self.reset_count = 0

    def reset(self):
        self.frames = []
        self.reset_count += 1

    def detect(self, frame, frame_id):
        self.frames.append((frame, frame_id))
        return [frame_id, frame_id, frame_id + 2, frame_id + 2]


def test_match_prediction_uses_pixel_distance_threshold():
    label = {"ball": {"visible": True, "x": 100.0, "y": 200.0}}

    assert match_prediction(label, (109.0, 200.0), threshold_px=10.0)["hit"] is True
    assert match_prediction(label, (111.0, 200.0), threshold_px=10.0)["hit"] is False


def test_compute_metrics_counts_hits_misses_and_false_positives():
    labels = [
        {"frame": 1, "ball": {"visible": True, "x": 100.0, "y": 100.0}},
        {"frame": 2, "ball": {"visible": True, "x": 200.0, "y": 200.0}},
        {"frame": 3, "ball": {"visible": False}},
    ]
    predictions = {
        1: (106.0, 100.0),
        2: None,
        3: (10.0, 10.0),
    }

    summary, rows = compute_metrics(labels, predictions, threshold_px=10.0)

    assert summary["visible_frames"] == 2
    assert summary["hits"] == 1
    assert summary["misses"] == 1
    assert summary["false_positive_frames"] == 1
    assert summary["precision"] == 0.5
    assert summary["recall"] == 0.5
    assert rows[0]["error_px"] == 6.0


def test_tracknet_prediction_uses_consecutive_temporal_window():
    detector = _FakeTemporalDetector()

    prediction = tracknet_prediction(_FakeCapture(), detector, frame_no=10)

    assert detector.reset_count == 1
    assert detector.frames == [(8, 8), (9, 9), (10, 10)]
    assert prediction == (11.0, 11.0)


def test_tracknet_prediction_skips_frames_without_enough_history():
    detector = _FakeTemporalDetector()

    assert tracknet_prediction(_FakeCapture(), detector, frame_no=1) is None
    assert detector.reset_count == 0
