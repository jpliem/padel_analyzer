"""
WallCollisionDetector — detect when a ball trajectory crosses a court wall.

Uses 3D ray-plane intersection (via PadelCourtModel) when z data is available,
falling back to 2D proximity otherwise.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from models.court_model import PadelCourtModel


class WallCollisionDetector:
    """
    Detect wall collisions using 3D ray-plane intersection or 2D proximity.

    Parameters
    ----------
    court_model : PadelCourtModel
        The 3D court geometry model to query.
    """

    def __init__(self, court_model: PadelCourtModel):
        self._court = court_model
        self._proximity_threshold_2d = 0.3

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(
        self,
        ball_pos: Optional[Dict],
        prev_ball_pos: Optional[Dict],
        use_3d: bool = True,
    ) -> Optional[Dict]:
        """
        Check if the ball trajectory crossed a wall.

        Parameters
        ----------
        ball_pos : dict or None
            Current ball position with keys x, y, z, speed.
        prev_ball_pos : dict or None
            Previous ball position with keys x, y, z, speed.
        use_3d : bool
            If True and any z > 0.01, perform 3D ray-plane intersection.
            Otherwise fall back to 2D proximity.

        Returns
        -------
        dict with wall_id, surface_type, impact_point, speed_at_impact,
        incoming_angle — or None if no collision detected.
        """
        if ball_pos is None or prev_ball_pos is None:
            return None

        x = float(ball_pos["x"])
        y = float(ball_pos["y"])
        z = float(ball_pos.get("z", 0.0))
        speed = float(ball_pos.get("speed", 0.0))

        prev_z = float(prev_ball_pos.get("z", 0.0))

        if use_3d and (z > 0.01 or prev_z > 0.01):
            p1 = (x, y, z)
            p2 = (
                float(prev_ball_pos["x"]),
                float(prev_ball_pos["y"]),
                prev_z,
            )
            return self._check_3d(p1, p2, speed)
        else:
            return self._check_2d(x, y, speed)

    def reset(self) -> None:
        """No-op — detector is stateless."""
        pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_3d(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        speed: float,
    ) -> Optional[Dict]:
        """
        Perform 3D ray-plane intersection from p2 (previous) toward p1 (current).

        The ray is cast from the previous position through the current position
        so we detect whether the trajectory crossed a wall in this frame step.
        """
        result = self._court.ray_intersect_walls(p2, p1)
        if result is None:
            return None

        # Allow hits where the impact point is close to the current ball
        # position, not just strictly within the segment.  The ray is cast
        # from prev (p2) toward current (p1); t=1 lands exactly at p1.
        # We accept the hit when the Euclidean distance from the current
        # position to the impact point is within 1 metre (i.e. wall is
        # imminent or just crossed).
        impact = result["point"]
        dist_to_impact = math.sqrt(
            (impact[0] - p1[0]) ** 2
            + (impact[1] - p1[1]) ** 2
            + (impact[2] - p1[2]) ** 2
        )
        if result["t"] > 1.0 + 1e-9 and dist_to_impact > 1.0:
            return None

        direction = (
            p1[0] - p2[0],
            p1[1] - p2[1],
            p1[2] - p2[2],
        )
        wall = self._court.get_wall_by_id(result["wall_id"])
        normal = wall.plane_normal if wall is not None else (0.0, 1.0, 0.0)

        incoming_angle = self._compute_angle(direction, normal)

        return {
            "wall_id": result["wall_id"],
            "surface_type": result["surface_type"],
            "impact_point": result["point"],
            "speed_at_impact": speed,
            "incoming_angle": incoming_angle,
        }

    def _check_2d(
        self,
        x: float,
        y: float,
        speed: float,
    ) -> Optional[Dict]:
        """
        2D proximity fallback — used when 3D data is unavailable.

        Queries the court model for the nearest wall within the proximity
        threshold at z=0.5 (typical ball height for 2D fallback).
        """
        wall = self._court.nearest_wall(x, y, 0.5, threshold=self._proximity_threshold_2d)
        if wall is None:
            return None

        # Best-effort impact point: project position onto the wall plane
        nx, ny, nz = wall.plane_normal
        ppx, ppy, ppz = wall.plane_point
        signed_dist = nx * (x - ppx) + ny * (y - ppy) + nz * (0.5 - ppz)
        impact_point = (
            x - signed_dist * nx,
            y - signed_dist * ny,
            0.5 - signed_dist * nz,
        )

        return {
            "wall_id": wall.wall_id,
            "surface_type": wall.surface_type,
            "impact_point": impact_point,
            "speed_at_impact": speed,
            "incoming_angle": 0.0,  # unknown without trajectory
        }

    @staticmethod
    def _compute_angle(
        direction: Tuple[float, float, float],
        normal: Tuple[float, float, float],
    ) -> float:
        """
        Compute the angle between the ball trajectory and the wall normal.

        Returns the angle in degrees between the direction vector and the
        wall's inward normal. A head-on hit (perpendicular to wall) gives 0°.

        Parameters
        ----------
        direction : (dx, dy, dz)
            Ball travel direction vector (unnormalized).
        normal : (nx, ny, nz)
            Wall unit normal vector.

        Returns
        -------
        float : angle in degrees [0, 90]
        """
        dx, dy, dz = direction
        nx, ny, nz = normal

        dir_mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        norm_mag = math.sqrt(nx * nx + ny * ny + nz * nz)

        if dir_mag < 1e-9 or norm_mag < 1e-9:
            return 0.0

        dot = abs(dx * nx + dy * ny + dz * nz)
        cos_angle = dot / (dir_mag * norm_mag)

        # Clamp to [-1, 1] to guard against floating-point drift
        cos_angle = max(-1.0, min(1.0, cos_angle))

        return math.degrees(math.acos(cos_angle))
