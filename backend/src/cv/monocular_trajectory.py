"""Gravity-constrained 3D trajectory fitting from one calibrated camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


GRAVITY = np.array([0.0, 0.0, -9.81], dtype=float)


@dataclass(frozen=True)
class RayObservation:
    timestamp: float
    camera_origin: Tuple[float, float, float]
    ray_direction: Tuple[float, float, float]
    confidence: float = 1.0
    is_ground_contact: bool = False
    frame_number: int = 0


@dataclass(frozen=True)
class FittedTrajectoryPoint:
    timestamp: float
    x: float
    y: float
    z: float
    frame_number: int = 0


@dataclass(frozen=True)
class MonocularTrajectoryFit:
    initial_position: Tuple[float, float, float]
    initial_velocity: Tuple[float, float, float]
    points: Tuple[FittedTrajectoryPoint, ...]
    median_ray_error_m: float
    max_ray_error_m: float
    condition_number: float
    confidence: float
    reliable: bool


class MonocularTrajectoryEstimator:
    """Fit one ballistic flight segment to a temporal sequence of image rays."""

    def __init__(self, gravity: Sequence[float] = GRAVITY,
                 max_reliable_error_m: float = 0.35,
                 max_condition_number: float = 1e8,
                 court_bounds_x: Tuple[float, float] = (-2.0, 12.0),
                 court_bounds_y: Tuple[float, float] = (-2.0, 22.0),
                 max_reliable_z: float = 12.0) -> None:
        self.gravity = np.asarray(gravity, dtype=float)
        self.max_reliable_error_m = max_reliable_error_m
        self.max_condition_number = max_condition_number
        # A geometrically consistent fit can still be a wrong-object track
        # (racket, head, reflection). Fits outside the playable volume are
        # never trusted, no matter how well the rays agree.
        self.court_bounds_x = court_bounds_x
        self.court_bounds_y = court_bounds_y
        self.max_reliable_z = max_reliable_z

    def fit(self, observations: Iterable[RayObservation]) -> Optional[MonocularTrajectoryFit]:
        obs = sorted(observations, key=lambda item: item.timestamp)
        if len(obs) < 3:
            return None
        t0 = obs[0].timestamp
        if obs[-1].timestamp - t0 <= 1e-6:
            return None

        rows: List[np.ndarray] = []
        targets: List[np.ndarray] = []
        valid_obs: List[RayObservation] = []
        for item in obs:
            direction = np.asarray(item.ray_direction, dtype=float)
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm <= 1e-9:
                continue
            direction /= norm
            origin = np.asarray(item.camera_origin, dtype=float)
            dt = item.timestamp - t0
            projector = np.eye(3) - np.outer(direction, direction)
            weight = max(0.05, min(1.0, item.confidence)) ** 0.5
            rows.append(np.hstack((projector, projector * dt)) * weight)
            targets.append(projector @ (origin - 0.5 * self.gravity * dt * dt) * weight)
            valid_obs.append(item)

            if item.is_ground_contact:
                rows.append(np.array([[0.0, 0.0, 1.0, 0.0, 0.0, dt]]) * (2.0 * weight))
                targets.append(np.array([-0.5 * self.gravity[2] * dt * dt]) * (2.0 * weight))

        if len(valid_obs) < 3:
            return None
        matrix = np.vstack(rows)
        target = np.concatenate([np.atleast_1d(value) for value in targets])
        solution, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
        if rank < 6 or len(singular) < 6 or singular[-1] <= 1e-12:
            return None

        p0, v0 = solution[:3], solution[3:]
        points: List[FittedTrajectoryPoint] = []
        ray_errors: List[float] = []
        positive_depths = 0
        for item in valid_obs:
            dt = item.timestamp - t0
            position = p0 + v0 * dt + 0.5 * self.gravity * dt * dt
            direction = np.asarray(item.ray_direction, dtype=float)
            direction /= np.linalg.norm(direction)
            origin = np.asarray(item.camera_origin, dtype=float)
            along = float(np.dot(position - origin, direction))
            positive_depths += int(along > 0)
            nearest = origin + max(0.0, along) * direction
            ray_errors.append(float(np.linalg.norm(position - nearest)))
            points.append(FittedTrajectoryPoint(
                timestamp=item.timestamp, x=float(position[0]), y=float(position[1]),
                z=float(position[2]), frame_number=item.frame_number,
            ))

        median_error = float(np.median(ray_errors))
        max_error = float(np.max(ray_errors))
        condition = float(singular[0] / singular[-1])
        positive_ratio = positive_depths / len(valid_obs)
        error_score = max(0.0, 1.0 - median_error / max(self.max_reliable_error_m, 1e-6))
        condition_score = max(0.0, 1.0 - condition / self.max_condition_number)
        confidence = float(error_score * condition_score * positive_ratio)
        inside_playable_volume = all(
            self.court_bounds_x[0] <= point.x <= self.court_bounds_x[1]
            and self.court_bounds_y[0] <= point.y <= self.court_bounds_y[1]
            and point.z <= self.max_reliable_z
            for point in points
        )
        reliable = bool(
            median_error <= self.max_reliable_error_m
            and condition <= self.max_condition_number
            and positive_ratio >= 0.8
            and min(point.z for point in points) >= -0.35
            and inside_playable_volume
        )
        if not inside_playable_volume:
            confidence = 0.0
        return MonocularTrajectoryFit(
            initial_position=tuple(float(v) for v in p0),
            initial_velocity=tuple(float(v) for v in v0), points=tuple(points),
            median_ray_error_m=median_error, max_ray_error_m=max_error,
            condition_number=condition, confidence=confidence, reliable=reliable,
        )

