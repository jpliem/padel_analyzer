import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.score_from_3d import score_points


def test_score_points_rally_mode_emits_bounce_without_serve_gate():
    points = [
        {"t": 0.00, "x": 5.0, "y": 12.0, "z": 2.0, "on_court": True},
        {"t": 0.04, "x": 5.0, "y": 12.4, "z": 1.0, "on_court": True},
        {"t": 0.08, "x": 5.0, "y": 12.4, "z": 0.1, "on_court": True},
    ]

    result = score_points(points, first_server="near", mode="rally", fps=25.0)

    assert result["events"][0]["event_type"] == "BOUNCE"
    assert result["events"][0]["metadata"]["side"] == "far"
