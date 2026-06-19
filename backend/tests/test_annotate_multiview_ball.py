import numpy as np

from scripts.annotate_multiview_ball import frame_for_time, stack_views


def test_frame_for_time_applies_offset_and_fps():
    assert frame_for_time(10.0, fps=50.0, offset_seconds=0.0) == 500
    assert frame_for_time(10.0, fps=60.0, offset_seconds=1.0) == 540


def test_stack_views_resizes_to_same_height():
    left = np.zeros((100, 200, 3), dtype=np.uint8)
    right = np.zeros((50, 100, 3), dtype=np.uint8)

    out = stack_views(left, right, height=100)

    assert out.shape == (100, 400, 3)
