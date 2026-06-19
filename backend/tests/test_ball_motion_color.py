import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from scripts.ball_motion_color import ball_candidates


def _frames_with_shapes():
    prev = np.zeros((120, 160, 3), dtype=np.uint8)
    cur = prev.copy()
    nxt = prev.copy()
    yellow = (0, 255, 255)
    cv2.circle(cur, (40, 60), 5, yellow, -1)
    cv2.circle(nxt, (42, 60), 5, yellow, -1)
    cv2.ellipse(cur, (110, 60), (22, 5), 0, 0, 360, yellow, -1)
    cv2.ellipse(nxt, (112, 60), (22, 5), 0, 0, 360, yellow, -1)
    return prev, cur, nxt


def test_ball_candidates_reject_elongated_racket_like_blob():
    prev, cur, nxt = _frames_with_shapes()

    candidates = ball_candidates(
        prev,
        cur,
        nxt,
        lo=[20, 80, 120],
        hi=[40, 255, 255],
        amin=2.0,
        amax=400.0,
        min_circularity=0.55,
        min_fill_ratio=0.45,
    )

    assert len(candidates) == 1
    x, y, r = candidates[0][:3]
    assert x == 40.0
    assert y == 60.0
    assert 4.0 <= r <= 8.0
