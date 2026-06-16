"""Validate multi-camera triangulation: known 3D -> project to 2 cams -> recover."""
import numpy as np
import pytest

from cv.triangulation import triangulate, reprojection_errors


def _make_camera(position, look_at, fx=1500.0, w=1920, h=1080):
    """Build a 3x4 projection matrix P = K[R|t] for a pinhole camera.

    position/look_at in world (court) coordinates; Z is up.
    """
    position = np.asarray(position, dtype=np.float64)
    look_at = np.asarray(look_at, dtype=np.float64)

    forward = look_at - position
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)  # image-y points down

    # Rotation world->camera: rows are camera axes (right, down, forward)
    R = np.vstack([right, down, forward])
    t = -R @ position
    K = np.array([[fx, 0, w / 2.0], [0, fx, h / 2.0], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


# Two cameras viewing a 10x20m court from opposite corners, ~6m high.
CAM_A = _make_camera([-3, -3, 6], [5, 10, 0])
CAM_B = _make_camera([13, 23, 6], [5, 10, 0])


def _project(P, X):
    x = P @ np.append(X, 1.0)
    return x[0] / x[2], x[1] / x[2]


# Ball positions spanning the court at various heights.
BALL_POINTS = [
    (5.0, 10.0, 0.0),    # net center, ground
    (2.0, 5.0, 1.5),     # near-left, mid-air
    (8.0, 16.0, 3.0),    # far-right, high
    (5.0, 3.0, 0.2),     # near baseline, low
    (1.0, 18.0, 2.5),    # far-left, high
]


@pytest.mark.parametrize("truth", BALL_POINTS)
def test_triangulation_recovers_exact_point(truth):
    truth = np.array(truth)
    obs = [(CAM_A, _project(CAM_A, truth)), (CAM_B, _project(CAM_B, truth))]
    est = triangulate(obs)
    assert est is not None
    assert np.linalg.norm(est - truth) < 1e-6  # noise-free => exact


def test_triangulation_needs_two_views():
    truth = np.array([5.0, 10.0, 1.0])
    assert triangulate([(CAM_A, _project(CAM_A, truth))]) is None


def test_triangulation_robust_to_pixel_noise():
    """With ~2px detection noise, 3D error should stay well under ~15cm."""
    rng = np.random.default_rng(42)
    errors = []
    for truth in BALL_POINTS:
        truth = np.array(truth)
        for _ in range(50):
            pa = np.array(_project(CAM_A, truth)) + rng.normal(0, 2.0, 2)
            pb = np.array(_project(CAM_B, truth)) + rng.normal(0, 2.0, 2)
            est = triangulate([(CAM_A, tuple(pa)), (CAM_B, tuple(pb))])
            errors.append(np.linalg.norm(est - truth))
    median_err = float(np.median(errors))
    assert median_err < 0.15, f"median 3D error {median_err:.3f}m too high"


def test_reprojection_error_flags_bad_sync():
    """Mismatched observations (different 3D points) => large reprojection error."""
    pa = _project(CAM_A, np.array([2.0, 5.0, 1.5]))
    pb = _project(CAM_B, np.array([8.0, 16.0, 3.0]))  # different point => disagreement
    est = triangulate([(CAM_A, pa), (CAM_B, pb)])
    errs = reprojection_errors(est, [(CAM_A, pa), (CAM_B, pb)])
    assert max(errs) > 5.0  # views disagree -> high error -> low confidence
