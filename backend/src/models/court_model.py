"""
PadelCourtModel — 3D Court Geometry

Defines the standard padel court in 3D with named wall segments.

Coordinate system:
  X: 0..10  (court width, left to right when viewed from near baseline)
  Y: 0..20  (court length, near baseline=0, far baseline=20)
  Z: 0..up  (height above court surface)

Court layout:
  Net at Y=10
  Service lines at Y=6.95 (near) and Y=13.05 (far)
  Center line at X=5

Wall layout:
  back_near  (Y=0):  glass, back_wall_height high, X=0..10
  back_far   (Y=20): glass, back_wall_height high, X=0..10
  side_left_near  (X=0,  Y=0..side_glass_length):  glass, side_glass_height high
  side_left_far   (X=0,  Y=(20-side_glass_length)..20): glass, side_glass_height high
  side_right_near (X=10, Y=0..side_glass_length):  glass, side_glass_height high
  side_right_far  (X=10, Y=(20-side_glass_length)..20): glass, side_glass_height high
  fence_left  (X=0,  Y=side_glass_length..(20-side_glass_length)): fence
  fence_right (X=10, Y=side_glass_length..(20-side_glass_length)): fence
  [optional] mesh_near (Y=0, Z=back_wall_height..back_wall_height+mesh_height)
  [optional] mesh_far  (Y=20, Z=back_wall_height..back_wall_height+mesh_height)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class WallSegment:
    """A planar wall segment with surface type and 3D bounds."""
    wall_id: str
    surface_type: str  # "glass" | "fence" | "mesh"
    plane_point: Tuple[float, float, float]   # a point on the plane
    plane_normal: Tuple[float, float, float]  # unit inward-facing normal
    bounds_min: Tuple[float, float, float]    # (x_min, y_min, z_min)
    bounds_max: Tuple[float, float, float]    # (x_max, y_max, z_max)


# Defaults
_DEFAULTS: Dict[str, object] = {
    "court_width": 10.0,
    "court_length": 20.0,
    "net_y": 10.0,
    "service_line_near": 6.95,
    "service_line_far": 13.05,
    "back_wall_height": 4.0,
    "side_glass_height": 3.0,
    "side_glass_length": 4.0,
    "fence_height": 4.0,
    "include_mesh_above_back_walls": False,
    "mesh_height": 1.0,
}


def _normalize(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    norm = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (v[0] / norm, v[1] / norm, v[2] / norm)


class PadelCourtModel:
    """
    3D model of a padel court enclosure.

    Parameters
    ----------
    overrides : dict, optional
        Override any default court dimension or flag. Keys:
          back_wall_height, side_glass_height, side_glass_length,
          fence_height, mesh_height, include_mesh_above_back_walls,
          court_width, court_length
        Unknown keys are silently ignored.
    """

    def __init__(self, overrides: Optional[Dict] = None):
        cfg = dict(_DEFAULTS)
        if overrides:
            for k, v in overrides.items():
                if k in cfg:
                    cfg[k] = v
                # unknown keys silently ignored

        self.court_width: float = cfg["court_width"]
        self.court_length: float = cfg["court_length"]
        self.net_y: float = cfg["net_y"]
        self.service_line_near: float = cfg["service_line_near"]
        self.service_line_far: float = cfg["service_line_far"]
        self.center_x: float = self.court_width / 2.0

        self._back_wall_height: float = cfg["back_wall_height"]
        self._side_glass_height: float = cfg["side_glass_height"]
        self._side_glass_length: float = cfg["side_glass_length"]
        self._fence_height: float = cfg["fence_height"]
        self._include_mesh: bool = cfg["include_mesh_above_back_walls"]
        self._mesh_height: float = cfg["mesh_height"]

        self._walls: List[WallSegment] = self._build_walls()
        self._wall_index: Dict[str, WallSegment] = {w.wall_id: w for w in self._walls}

    # ------------------------------------------------------------------
    # Wall construction
    # ------------------------------------------------------------------

    def _build_walls(self) -> List[WallSegment]:
        W = self.court_width
        L = self.court_length
        bwh = self._back_wall_height
        sgh = self._side_glass_height
        sgl = self._side_glass_length
        fh = self._fence_height

        walls: List[WallSegment] = []

        # --- Back walls (glass) ---
        # Near back wall: Y=0, normal points +Y (inward)
        walls.append(WallSegment(
            wall_id="back_near",
            surface_type="glass",
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=_normalize((0.0, 1.0, 0.0)),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(W, 0.0, bwh),
        ))

        # Far back wall: Y=L, normal points -Y (inward)
        walls.append(WallSegment(
            wall_id="back_far",
            surface_type="glass",
            plane_point=(0.0, L, 0.0),
            plane_normal=_normalize((0.0, -1.0, 0.0)),
            bounds_min=(0.0, L, 0.0),
            bounds_max=(W, L, bwh),
        ))

        # --- Side glass sections ---
        # Left side (X=0), near section: Y=0..sgl
        walls.append(WallSegment(
            wall_id="side_left_near",
            surface_type="glass",
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=_normalize((1.0, 0.0, 0.0)),
            bounds_min=(0.0, 0.0, 0.0),
            bounds_max=(0.0, sgl, sgh),
        ))

        # Left side (X=0), far section: Y=(L-sgl)..L
        walls.append(WallSegment(
            wall_id="side_left_far",
            surface_type="glass",
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=_normalize((1.0, 0.0, 0.0)),
            bounds_min=(0.0, L - sgl, 0.0),
            bounds_max=(0.0, L, sgh),
        ))

        # Right side (X=W), near section: Y=0..sgl
        walls.append(WallSegment(
            wall_id="side_right_near",
            surface_type="glass",
            plane_point=(W, 0.0, 0.0),
            plane_normal=_normalize((-1.0, 0.0, 0.0)),
            bounds_min=(W, 0.0, 0.0),
            bounds_max=(W, sgl, sgh),
        ))

        # Right side (X=W), far section: Y=(L-sgl)..L
        walls.append(WallSegment(
            wall_id="side_right_far",
            surface_type="glass",
            plane_point=(W, 0.0, 0.0),
            plane_normal=_normalize((-1.0, 0.0, 0.0)),
            bounds_min=(W, L - sgl, 0.0),
            bounds_max=(W, L, sgh),
        ))

        # --- Fence sections (middle side sections) ---
        # Left fence: X=0, Y=sgl..(L-sgl)
        walls.append(WallSegment(
            wall_id="fence_left",
            surface_type="fence",
            plane_point=(0.0, 0.0, 0.0),
            plane_normal=_normalize((1.0, 0.0, 0.0)),
            bounds_min=(0.0, sgl, 0.0),
            bounds_max=(0.0, L - sgl, fh),
        ))

        # Right fence: X=W, Y=sgl..(L-sgl)
        walls.append(WallSegment(
            wall_id="fence_right",
            surface_type="fence",
            plane_point=(W, 0.0, 0.0),
            plane_normal=_normalize((-1.0, 0.0, 0.0)),
            bounds_min=(W, sgl, 0.0),
            bounds_max=(W, L - sgl, fh),
        ))

        # --- Optional mesh above back walls ---
        if self._include_mesh:
            mh = self._mesh_height
            walls.append(WallSegment(
                wall_id="mesh_near",
                surface_type="mesh",
                plane_point=(0.0, 0.0, 0.0),
                plane_normal=_normalize((0.0, 1.0, 0.0)),
                bounds_min=(0.0, 0.0, bwh),
                bounds_max=(W, 0.0, bwh + mh),
            ))
            walls.append(WallSegment(
                wall_id="mesh_far",
                surface_type="mesh",
                plane_point=(0.0, L, 0.0),
                plane_normal=_normalize((0.0, -1.0, 0.0)),
                bounds_min=(0.0, L, bwh),
                bounds_max=(W, L, bwh + mh),
            ))

        return walls

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_wall_segments(self) -> List[WallSegment]:
        """Return all wall segments."""
        return list(self._walls)

    def get_wall_by_id(self, wall_id: str) -> Optional[WallSegment]:
        """Return the wall with the given wall_id, or None."""
        return self._wall_index.get(wall_id)

    def get_bounds(self) -> Dict[str, float]:
        """Return court bounds dict with x_min, x_max, y_min, y_max."""
        return {
            "x_min": 0.0,
            "x_max": self.court_width,
            "y_min": 0.0,
            "y_max": self.court_length,
        }

    def nearest_wall(
        self,
        x: float,
        y: float,
        z: float,
        threshold: float,
    ) -> Optional[WallSegment]:
        """
        Return the nearest wall within threshold distance, or None.

        Distance is the perpendicular distance from the point to the wall's
        infinite plane, but the point must project within the wall's XYZ bounds
        (including Z bounds) for the wall to be considered a candidate.

        Parameters
        ----------
        x, y, z : float
            3D position of the point.
        threshold : float
            Maximum perpendicular distance to wall plane.

        Returns
        -------
        WallSegment or None
        """
        best_wall = None
        best_dist = float("inf")

        for wall in self._walls:
            dist = self._perpendicular_distance(x, y, z, wall)
            if dist is None or dist > threshold:
                continue
            if dist < best_dist:
                best_dist = dist
                best_wall = wall

        return best_wall

    def ray_intersect_walls(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
    ) -> Optional[Dict]:
        """
        Find the first wall intersected by the ray from p1 toward p2.

        The ray is parameterised as:  P(t) = p1 + t * (p2 - p1), t >= 0.
        Returns the intersection with smallest t > 0 whose intersection
        point falls within the wall's bounds.

        Parameters
        ----------
        p1, p2 : (x, y, z) tuples
            Ray origin and a second point defining ray direction.

        Returns
        -------
        dict with keys {wall_id, surface_type, point, t}, or None.
        """
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]

        best_t = float("inf")
        best_result = None

        for wall in self._walls:
            nx, ny, nz = wall.plane_normal
            ppx, ppy, ppz = wall.plane_point

            # Dot product of normal with ray direction
            denom = nx * dx + ny * dy + nz * dz

            # Parallel ray (or very nearly so) — no intersection
            if abs(denom) < 1e-9:
                continue

            # t = dot(normal, plane_point - ray_origin) / dot(normal, ray_dir)
            t = (
                nx * (ppx - p1[0]) + ny * (ppy - p1[1]) + nz * (ppz - p1[2])
            ) / denom

            if t < -1e-9:  # intersection behind ray origin
                continue

            # Compute intersection point
            ix = p1[0] + t * dx
            iy = p1[1] + t * dy
            iz = p1[2] + t * dz

            # Check whether intersection point is within wall bounds
            if not self._in_bounds(ix, iy, iz, wall):
                continue

            if t < best_t:
                best_t = t
                best_result = {
                    "wall_id": wall.wall_id,
                    "surface_type": wall.surface_type,
                    "point": (ix, iy, iz),
                    "t": t,
                }

        return best_result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _perpendicular_distance(
        self,
        x: float,
        y: float,
        z: float,
        wall: WallSegment,
    ) -> Optional[float]:
        """
        Perpendicular distance from point (x,y,z) to wall plane, but only
        if the point's projection onto the plane falls within wall bounds.

        Returns None if the projection is outside bounds or if Z is out of range.
        """
        nx, ny, nz = wall.plane_normal
        ppx, ppy, ppz = wall.plane_point

        # Signed distance along normal from plane
        signed_dist = nx * (x - ppx) + ny * (y - ppy) + nz * (z - ppz)
        dist = abs(signed_dist)

        # Project point onto the plane
        px = x - signed_dist * nx
        py = y - signed_dist * ny
        pz = z - signed_dist * nz

        # Check bounds of projection
        if not self._in_bounds(px, py, pz, wall):
            return None

        return dist

    @staticmethod
    def _in_bounds(
        x: float,
        y: float,
        z: float,
        wall: WallSegment,
        tol: float = 1e-6,
    ) -> bool:
        """Return True if (x, y, z) is within wall.bounds_min/max (with tolerance)."""
        bmin = wall.bounds_min
        bmax = wall.bounds_max

        # For walls aligned to a coordinate axis, the "thin" dimension has
        # bounds_min == bounds_max. We skip checking that dimension
        # (or allow a small tolerance around the plane value).
        checks = []
        for val, lo, hi in ((x, bmin[0], bmax[0]), (y, bmin[1], bmax[1]), (z, bmin[2], bmax[2])):
            if abs(hi - lo) < 1e-9:
                # Degenerate (plane) dimension — check proximity to plane
                checks.append(abs(val - lo) <= tol + 1e-6)
            else:
                checks.append(lo - tol <= val <= hi + tol)

        return all(checks)
