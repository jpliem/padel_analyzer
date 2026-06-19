import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.triangulate_matched import (
    choose_best_match,
    clean_3d_track,
    filter_candidates_near_boxes,
    filter_player_boxes,
)


def _make_camera(position, look_at, fx=1500.0, w=1920, h=1080):
    position = np.asarray(position, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)
    forward = look_at - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.vstack([right, down, forward])
    t = -R @ position
    K = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


def _project(P, X):
    x = P @ np.append(np.asarray(X, dtype=np.float64), 1.0)
    return x[0] / x[2], x[1] / x[2]


CAM_A = _make_camera([-3, -3, 6], [5, 10, 0])
CAM_B = _make_camera([13, 23, 6], [5, 10, 0])


def _candidate_pair(point):
    xa, ya = _project(CAM_A, point)
    xb, yb = _project(CAM_B, point)
    return (xa, ya, 10.0), (xb, yb, 10.0)


def test_choose_best_match_prefers_temporal_continuity_over_equal_geometry():
    near_a, near_b = _candidate_pair([5.3, 10.2, 1.1])
    far_a, far_b = _candidate_pair([1.0, 18.0, 2.0])

    match = choose_best_match(
        [far_a, near_a],
        [far_b, near_b],
        CAM_A,
        CAM_B,
        max_reproj=1.0,
        previous_point={"x": 5.0, "y": 10.0, "z": 1.0},
        continuity_weight=3.0,
    )

    assert match is not None
    assert match["x"] == 5.3
    assert match["y"] == 10.2
    assert match["z"] == 1.1


def test_clean_3d_track_rejects_teleports_and_interpolates_short_gaps():
    raw = [
        {"t": 0.00, "x": 5.0, "y": 10.0, "z": 1.0, "reproj_px": 3.0, "on_court": True},
        {"t": 0.04, "x": 5.2, "y": 10.1, "z": 1.1, "reproj_px": 3.0, "on_court": True},
        {"t": 0.08, "x": 0.0, "y": 20.0, "z": 4.0, "reproj_px": 2.0, "on_court": True},
        {"t": 0.16, "x": 5.6, "y": 10.3, "z": 1.2, "reproj_px": 3.0, "on_court": True},
    ]

    clean = clean_3d_track(raw, max_speed_mps=20.0, interpolate_rate=25.0, max_interpolate_gap=0.2)

    assert [p["t"] for p in clean] == [0.0, 0.04, 0.08, 0.12, 0.16]
    assert all(4.9 <= p["x"] <= 5.7 for p in clean)
    assert all(9.9 <= p["y"] <= 10.4 for p in clean)
    assert all(p["on_court"] for p in clean)


def test_clean_3d_track_does_not_interpolate_long_sparse_gaps_by_default():
    raw = [
        {"t": 2.50, "x": 1.52, "y": 15.4, "z": 1.73, "reproj_px": 3.0, "on_court": True},
        {"t": 2.90, "x": 5.95, "y": 2.02, "z": 1.59, "reproj_px": 4.8, "on_court": True},
    ]

    clean = clean_3d_track(raw, max_speed_mps=45.0, interpolate_rate=25.0)

    assert [p["t"] for p in clean] == [2.5, 2.9]


def test_filter_candidates_near_boxes_rejects_player_and_racket_region():
    candidates = [
        {"x": 100.0, "y": 100.0, "r": 4.0},
        {"x": 155.0, "y": 120.0, "r": 4.0},
        {"x": 240.0, "y": 120.0, "r": 4.0},
    ]
    boxes = [[90.0, 80.0, 150.0, 220.0]]

    kept = filter_candidates_near_boxes(candidates, boxes, x_margin_frac=0.5, y_margin_frac=0.1)

    assert kept == [{"x": 240.0, "y": 120.0, "r": 4.0}]


def test_filter_player_boxes_rejects_large_wall_mural_person_box():
    boxes = [
        [100.0, 600.0, 220.0, 950.0],
        [1900.0, 5.0, 2500.0, 635.0],
    ]

    kept = filter_player_boxes(boxes, image_shape=(1080, 3840, 3))

    assert kept == [[100.0, 600.0, 220.0, 950.0]]
