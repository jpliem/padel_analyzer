"""Multi-camera triangulation — recover a 3D point from 2+ camera views.

Each calibrated camera gives a 3x4 projection matrix P (= K[R|t]) and a pixel
observation of the same world point. Two or more views constrain the 3D point
geometrically (ray intersection), with no dependence on a noisy single-camera
height estimate — this is the robust path to true ball x,y,z.

Uses the linear DLT (Direct Linear Transform): each view contributes two rows
to a homogeneous system A·X = 0; the solution is the right singular vector of
A for the smallest singular value.
"""
from typing import List, Tuple, Optional
import numpy as np


def triangulate(observations: List[Tuple[np.ndarray, Tuple[float, float]]]) -> Optional[np.ndarray]:
    """Triangulate one 3D point from N≥2 (projection_matrix, pixel) observations.

    Args:
        observations: list of (P, (px, py)); P is a 3x4 projection matrix.

    Returns:
        np.array([x, y, z]) world point, or None if fewer than 2 views.
    """
    if len(observations) < 2:
        return None

    rows = []
    for P, (px, py) in observations:
        P = np.asarray(P, dtype=np.float64)
        # px * (P row3) - (P row1) = 0  ;  py * (P row3) - (P row2) = 0
        rows.append(px * P[2, :] - P[0, :])
        rows.append(py * P[2, :] - P[1, :])

    A = np.asarray(rows, dtype=np.float64)
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def reprojection_errors(point_3d: np.ndarray,
                        observations: List[Tuple[np.ndarray, Tuple[float, float]]]) -> List[float]:
    """Per-view reprojection error (pixels) for a triangulated point.

    Useful as a confidence/quality signal — a large error means the views
    disagree (bad sync, bad detection, or bad calibration).
    """
    errs = []
    X = np.append(np.asarray(point_3d, dtype=np.float64), 1.0)
    for P, (px, py) in observations:
        proj = np.asarray(P, dtype=np.float64) @ X
        if abs(proj[2]) < 1e-12:
            errs.append(float("inf"))
            continue
        u, v = proj[0] / proj[2], proj[1] / proj[2]
        errs.append(float(np.hypot(u - px, v - py)))
    return errs
