import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.eval_ball_labels import compute_metrics, match_prediction


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
