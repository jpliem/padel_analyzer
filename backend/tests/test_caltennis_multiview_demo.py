import os
import sys

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.caltennis_multiview_demo import choose_pair, tracknet_candidate


def _camera(center, target):
    center = np.asarray(center, dtype=float)
    forward = np.asarray(target, dtype=float) - center
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack([right, down, forward])
    translation = -rotation @ center
    intrinsic = np.array([[1200, 0, 960], [0, 1200, 540], [0, 0, 1]])
    return intrinsic @ np.hstack([rotation, translation[:, None]])


def _project(matrix, point):
    pixel = matrix @ np.append(point, 1.0)
    return pixel[:2] / pixel[2]


def test_choose_pair_recovers_geometrically_consistent_candidate():
    projection_a = _camera([5, -8, 4], [5, 10, 1])
    projection_b = _camera([5, 28, 4], [5, 10, 1])
    point = np.array([6.0, 12.0, 1.4])
    pixel_a = _project(projection_a, point)
    pixel_b = _project(projection_b, point)
    candidates_a = [{"x": pixel_a[0], "y": pixel_a[1]}, {"x": 100, "y": 100}]
    candidates_b = [{"x": 1500, "y": 800}, {"x": pixel_b[0], "y": pixel_b[1]}]

    match = choose_pair(candidates_a, candidates_b, projection_a, projection_b, 1.0)

    assert match is not None
    assert np.allclose(match["point"], point)
    assert max(match["errors"]) < 1e-8


def test_tracknet_candidate_converts_box_to_center():
    class Detector:
        def reset(self):
            self.calls = 0

        def detect(self, frame, frame_id):
            self.calls += 1
            return None if self.calls < 3 else [10, 20, 18, 32]

    result = tracknet_candidate(Detector(), [object(), object(), object()])

    assert result == [{"x": 14.0, "y": 26.0, "r": 4.0, "source": "tracknet"}]
