"""
TDD tests for PadelCourtModel — 3D Court Geometry.

Standard padel court:
  X: 0..10 (width, left to right)
  Y: 0..20 (length, one baseline to other)
  Z: 0..4+ (height)
  Net at Y=10
  Service lines at Y=6.95 and Y=13.05
  Center line at X=5

Wall layout:
  Back walls (glass, 4m high): Y=0 and Y=20, full width X=0..10
  Side glass sections (3m high): extend 4m inward from each back wall
    - Left side (X=0):  Y=0..4 and Y=16..20  (glass, 3m high)
    - Right side (X=10): Y=0..4 and Y=16..20 (glass, 3m high)
  Side fence sections (fence, height ~4m): middle section
    - Left side (X=0):  Y=4..16
    - Right side (X=10): Y=4..16
  Optional mesh above back walls: Y=0 and Y=20, Z=4..5 (1m mesh)
"""

import math
import pytest
import numpy as np

from models.court_model import PadelCourtModel, WallSegment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def court():
    return PadelCourtModel()


@pytest.fixture
def court_custom():
    """Court with overridden back_wall_height."""
    return PadelCourtModel(overrides={"back_wall_height": 3.5})


@pytest.fixture
def court_with_mesh():
    return PadelCourtModel(overrides={"include_mesh_above_back_walls": True})


# ---------------------------------------------------------------------------
# 1. Default dimensions
# ---------------------------------------------------------------------------

class TestDefaultDimensions:
    def test_width(self, court):
        b = court.get_bounds()
        assert b["x_min"] == pytest.approx(0.0)
        assert b["x_max"] == pytest.approx(10.0)

    def test_length(self, court):
        b = court.get_bounds()
        assert b["y_min"] == pytest.approx(0.0)
        assert b["y_max"] == pytest.approx(20.0)

    def test_net_position(self, court):
        assert court.net_y == pytest.approx(10.0)

    def test_service_lines(self, court):
        assert court.service_line_near == pytest.approx(6.95)
        assert court.service_line_far == pytest.approx(13.05)

    def test_center_x(self, court):
        assert court.center_x == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# 2. Wall segments presence and counts
# ---------------------------------------------------------------------------

class TestWallSegments:
    def test_returns_list(self, court):
        walls = court.get_wall_segments()
        assert isinstance(walls, list)
        assert len(walls) > 0

    def test_each_wall_is_wall_segment(self, court):
        for w in court.get_wall_segments():
            assert isinstance(w, WallSegment)

    def test_back_walls_present(self, court):
        walls = court.get_wall_segments()
        back = [w for w in walls if "back" in w.wall_id]
        assert len(back) == 2

    def test_side_glass_sections(self, court):
        walls = court.get_wall_segments()
        side_glass = [w for w in walls if w.surface_type == "glass" and "side" in w.wall_id]
        # 4 side glass sections: left near, left far, right near, right far
        assert len(side_glass) == 4

    def test_side_fence_sections(self, court):
        walls = court.get_wall_segments()
        fence = [w for w in walls if w.surface_type == "fence"]
        # 2 fence sections: left middle, right middle
        assert len(fence) == 2

    def test_no_mesh_by_default(self, court):
        walls = court.get_wall_segments()
        mesh = [w for w in walls if w.surface_type == "mesh"]
        assert len(mesh) == 0

    def test_mesh_present_when_enabled(self, court_with_mesh):
        walls = court_with_mesh.get_wall_segments()
        mesh = [w for w in walls if w.surface_type == "mesh"]
        assert len(mesh) == 2  # one above each back wall

    def test_unique_wall_ids(self, court):
        walls = court.get_wall_segments()
        ids = [w.wall_id for w in walls]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# 3. WallSegment fields
# ---------------------------------------------------------------------------

class TestWallSegmentFields:
    def test_required_fields_present(self, court):
        for w in court.get_wall_segments():
            assert hasattr(w, "wall_id")
            assert hasattr(w, "surface_type")
            assert hasattr(w, "plane_point")
            assert hasattr(w, "plane_normal")
            assert hasattr(w, "bounds_min")
            assert hasattr(w, "bounds_max")

    def test_surface_types_valid(self, court):
        valid = {"glass", "fence", "mesh"}
        for w in court.get_wall_segments():
            assert w.surface_type in valid

    def test_plane_point_is_3d(self, court):
        for w in court.get_wall_segments():
            assert len(w.plane_point) == 3

    def test_plane_normal_is_3d(self, court):
        for w in court.get_wall_segments():
            assert len(w.plane_normal) == 3

    def test_plane_normal_is_unit_vector(self, court):
        for w in court.get_wall_segments():
            n = np.array(w.plane_normal, dtype=float)
            assert np.linalg.norm(n) == pytest.approx(1.0, abs=1e-6)

    def test_bounds_min_max_3d(self, court):
        for w in court.get_wall_segments():
            assert len(w.bounds_min) == 3
            assert len(w.bounds_max) == 3

    def test_bounds_min_le_max(self, court):
        for w in court.get_wall_segments():
            for i in range(3):
                assert w.bounds_min[i] <= w.bounds_max[i], (
                    f"Wall {w.wall_id} bounds_min[{i}]={w.bounds_min[i]} > bounds_max[{i}]={w.bounds_max[i]}"
                )


# ---------------------------------------------------------------------------
# 4. Back wall geometry
# ---------------------------------------------------------------------------

class TestBackWallGeometry:
    def _get_back_wall(self, court, side):
        """side: 'near' (Y=0) or 'far' (Y=20)"""
        walls = court.get_wall_segments()
        back = [w for w in walls if "back" in w.wall_id and side in w.wall_id]
        assert len(back) == 1, f"Expected 1 {side} back wall, got {len(back)}"
        return back[0]

    def test_near_back_wall_surface_type(self, court):
        w = self._get_back_wall(court, "near")
        assert w.surface_type == "glass"

    def test_far_back_wall_surface_type(self, court):
        w = self._get_back_wall(court, "far")
        assert w.surface_type == "glass"

    def test_near_back_wall_height(self, court):
        w = self._get_back_wall(court, "near")
        assert w.bounds_max[2] == pytest.approx(4.0)

    def test_far_back_wall_height(self, court):
        w = self._get_back_wall(court, "far")
        assert w.bounds_max[2] == pytest.approx(4.0)

    def test_near_back_wall_width(self, court):
        w = self._get_back_wall(court, "near")
        assert w.bounds_min[0] == pytest.approx(0.0)
        assert w.bounds_max[0] == pytest.approx(10.0)

    def test_near_back_wall_normal_inward(self, court):
        """Near back wall (Y=0) normal should point in +Y direction."""
        w = self._get_back_wall(court, "near")
        n = np.array(w.plane_normal)
        assert n[1] > 0, f"Near back wall normal should point +Y, got {n}"

    def test_far_back_wall_normal_inward(self, court):
        """Far back wall (Y=20) normal should point in -Y direction."""
        w = self._get_back_wall(court, "far")
        n = np.array(w.plane_normal)
        assert n[1] < 0, f"Far back wall normal should point -Y, got {n}"

    def test_override_back_wall_height(self, court_custom):
        walls = court_custom.get_wall_segments()
        back = [w for w in walls if "back" in w.wall_id]
        for w in back:
            assert w.bounds_max[2] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# 5. Side wall geometry
# ---------------------------------------------------------------------------

class TestSideWallGeometry:
    def test_left_glass_sections_at_x0(self, court):
        walls = court.get_wall_segments()
        left_glass = [
            w for w in walls
            if w.surface_type == "glass" and "side" in w.wall_id and "left" in w.wall_id
        ]
        assert len(left_glass) == 2
        for w in left_glass:
            # The wall plane is at X=0
            assert w.plane_point[0] == pytest.approx(0.0)

    def test_right_glass_sections_at_x10(self, court):
        walls = court.get_wall_segments()
        right_glass = [
            w for w in walls
            if w.surface_type == "glass" and "side" in w.wall_id and "right" in w.wall_id
        ]
        assert len(right_glass) == 2
        for w in right_glass:
            assert w.plane_point[0] == pytest.approx(10.0)

    def test_left_glass_height_3m(self, court):
        walls = court.get_wall_segments()
        left_glass = [
            w for w in walls
            if w.surface_type == "glass" and "side" in w.wall_id and "left" in w.wall_id
        ]
        for w in left_glass:
            assert w.bounds_max[2] == pytest.approx(3.0)

    def test_side_glass_extends_4m_from_back_wall(self, court):
        walls = court.get_wall_segments()
        left_near_glass = [
            w for w in walls
            if w.surface_type == "glass" and "side" in w.wall_id
            and "left" in w.wall_id and "near" in w.wall_id
        ]
        assert len(left_near_glass) == 1
        w = left_near_glass[0]
        # near side: Y=0..4
        assert w.bounds_min[1] == pytest.approx(0.0)
        assert w.bounds_max[1] == pytest.approx(4.0)

    def test_left_side_normal_inward(self, court):
        """Left side walls (X=0) normal should point in +X direction."""
        walls = court.get_wall_segments()
        left_walls = [w for w in walls if "left" in w.wall_id and "side" in w.wall_id]
        for w in left_walls:
            n = np.array(w.plane_normal)
            assert n[0] > 0, f"Left side wall normal should point +X, got {n}"

    def test_right_side_normal_inward(self, court):
        """Right side walls (X=10) normal should point in -X direction."""
        walls = court.get_wall_segments()
        right_walls = [w for w in walls if "right" in w.wall_id and "side" in w.wall_id]
        for w in right_walls:
            n = np.array(w.plane_normal)
            assert n[0] < 0, f"Right side wall normal should point -X, got {n}"

    def test_fence_middle_section_y_range(self, court):
        walls = court.get_wall_segments()
        left_fence = [w for w in walls if w.surface_type == "fence" and "left" in w.wall_id]
        assert len(left_fence) == 1
        w = left_fence[0]
        assert w.bounds_min[1] == pytest.approx(4.0)
        assert w.bounds_max[1] == pytest.approx(16.0)


# ---------------------------------------------------------------------------
# 6. get_wall_by_id
# ---------------------------------------------------------------------------

class TestGetWallById:
    def test_returns_correct_wall(self, court):
        walls = court.get_wall_segments()
        for w in walls:
            found = court.get_wall_by_id(w.wall_id)
            assert found is not None
            assert found.wall_id == w.wall_id

    def test_unknown_id_returns_none(self, court):
        assert court.get_wall_by_id("nonexistent_wall") is None


# ---------------------------------------------------------------------------
# 7. nearest_wall
# ---------------------------------------------------------------------------

class TestNearestWall:
    def test_point_near_near_back_wall(self, court):
        # Ball at X=5, Y=0.1, Z=1.5 — very close to near back wall
        result = court.nearest_wall(5.0, 0.1, 1.5, threshold=1.0)
        assert result is not None
        assert "back" in result.wall_id
        assert "near" in result.wall_id

    def test_point_near_far_back_wall(self, court):
        result = court.nearest_wall(5.0, 19.9, 1.5, threshold=1.0)
        assert result is not None
        assert "back" in result.wall_id
        assert "far" in result.wall_id

    def test_point_near_left_side(self, court):
        # Ball at X=0.1, Y=2 (within side glass section), Z=1.5
        result = court.nearest_wall(0.1, 2.0, 1.5, threshold=1.0)
        assert result is not None
        assert "left" in result.wall_id

    def test_point_near_right_side(self, court):
        result = court.nearest_wall(9.9, 2.0, 1.5, threshold=1.0)
        assert result is not None
        assert "right" in result.wall_id

    def test_point_far_from_all_walls_returns_none(self, court):
        # Center of the court
        result = court.nearest_wall(5.0, 10.0, 1.0, threshold=0.5)
        assert result is None

    def test_threshold_respected(self, court):
        # Ball at Y=0.8 — within 1m but not within 0.5m of near back wall
        result_far = court.nearest_wall(5.0, 0.8, 1.5, threshold=0.5)
        assert result_far is None
        result_near = court.nearest_wall(5.0, 0.3, 1.5, threshold=0.5)
        assert result_near is not None

    def test_ball_above_wall_height_not_matched(self, court):
        # Ball at Z=5.0 — above all walls (max 4m), should not match
        result = court.nearest_wall(5.0, 0.1, 5.0, threshold=1.0)
        assert result is None

    def test_returns_closest_wall(self, court):
        # Ball at corner X=0.1, Y=0.5, Z=1.0 — near both left side glass and near back wall
        # Y distance to near back = 0.5, X distance to left wall = 0.1
        # Should return left side wall (closer)
        result = court.nearest_wall(0.1, 0.5, 1.0, threshold=1.0)
        assert result is not None
        assert "left" in result.wall_id


# ---------------------------------------------------------------------------
# 8. ray_intersect_walls
# ---------------------------------------------------------------------------

class TestRayIntersectWalls:
    def test_ray_hits_near_back_wall(self, court):
        # Ray from center going toward near back wall (Y decreasing)
        p1 = (5.0, 5.0, 1.0)
        p2 = (5.0, -1.0, 1.0)  # direction: toward Y=0
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        assert "back" in result["wall_id"]
        assert "near" in result["wall_id"]
        assert "point" in result
        assert "t" in result
        assert "surface_type" in result
        assert result["t"] > 0

    def test_ray_hits_far_back_wall(self, court):
        p1 = (5.0, 15.0, 1.0)
        p2 = (5.0, 21.0, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        assert "back" in result["wall_id"]
        assert "far" in result["wall_id"]

    def test_ray_hits_left_side_glass(self, court):
        # Ray from middle going toward left glass (near side Y=2)
        p1 = (5.0, 2.0, 1.0)
        p2 = (-1.0, 2.0, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        assert "left" in result["wall_id"]

    def test_ray_hits_right_side_fence(self, court):
        # Ray going toward right fence (Y=10 is middle fence section)
        p1 = (5.0, 10.0, 1.0)
        p2 = (11.0, 10.0, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        assert "right" in result["wall_id"]
        assert result["surface_type"] == "fence"

    def test_ray_parallel_to_wall_returns_none(self, court):
        # Ray parallel to near back wall (moving in X only at Y=0.5)
        p1 = (2.0, 0.5, 1.0)
        p2 = (8.0, 0.5, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        # Should not intersect any back/side wall while moving purely in X
        # (may intersect side walls at X=0 or X=10 if extended, but direction is +X so no left wall hit)
        # The right side wall may be hit, but not the back wall
        if result is not None:
            assert "back" not in result["wall_id"] or "near" not in result["wall_id"]

    def test_ray_result_has_correct_keys(self, court):
        p1 = (5.0, 5.0, 1.0)
        p2 = (5.0, -1.0, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        for key in ("wall_id", "surface_type", "point", "t"):
            assert key in result, f"Missing key: {key}"

    def test_ray_intersection_point_on_wall_plane(self, court):
        p1 = (5.0, 5.0, 1.0)
        p2 = (5.0, -1.0, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        # Intersection point should be at Y=0 (near back wall)
        pt = result["point"]
        assert pt[1] == pytest.approx(0.0, abs=1e-6)

    def test_ray_no_intersection_going_away(self, court):
        # Ray starting in court, going upward (Z direction) — misses all walls
        p1 = (5.0, 10.0, 1.0)
        p2 = (5.0, 10.0, 10.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is None

    def test_ray_returns_closest_intersection(self, court):
        # Ray from near back wall toward far back wall — should hit near back wall first?
        # Actually starting inside court, going toward far wall — hits far wall
        p1 = (5.0, 1.0, 1.0)
        p2 = (5.0, 25.0, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        assert "far" in result["wall_id"]
        # t should be positive and minimal
        assert result["t"] > 0

    def test_ray_outside_wall_bounds_no_hit(self, court):
        # Ray going toward Y=20 (far back wall) but at Z=5 (above wall height of 4m)
        p1 = (5.0, 15.0, 5.0)
        p2 = (5.0, 25.0, 5.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is None

    def test_t_value_correct(self, court):
        # Ray from Y=5 going to Y=-5 (direction magnitude = 10 in Y)
        # Should hit near back wall (Y=0) at t=0.5 (5 units of 10 unit direction)
        p1 = (5.0, 5.0, 1.0)
        p2 = (5.0, -5.0, 1.0)
        result = court.ray_intersect_walls(p1, p2)
        assert result is not None
        assert result["t"] == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# 9. Overrides
# ---------------------------------------------------------------------------

class TestOverrides:
    def test_back_wall_height_override(self):
        court = PadelCourtModel(overrides={"back_wall_height": 3.5})
        walls = court.get_wall_segments()
        back = [w for w in walls if "back" in w.wall_id]
        for w in back:
            assert w.bounds_max[2] == pytest.approx(3.5)

    def test_side_glass_height_override(self):
        court = PadelCourtModel(overrides={"side_glass_height": 2.5})
        walls = court.get_wall_segments()
        side_glass = [w for w in walls if w.surface_type == "glass" and "side" in w.wall_id]
        for w in side_glass:
            assert w.bounds_max[2] == pytest.approx(2.5)

    def test_side_glass_length_override(self):
        court = PadelCourtModel(overrides={"side_glass_length": 3.0})
        walls = court.get_wall_segments()
        left_near_glass = [
            w for w in walls
            if w.surface_type == "glass" and "side" in w.wall_id
            and "left" in w.wall_id and "near" in w.wall_id
        ]
        assert len(left_near_glass) == 1
        w = left_near_glass[0]
        assert w.bounds_max[1] == pytest.approx(3.0)

    def test_unknown_override_ignored(self):
        # Should not raise
        court = PadelCourtModel(overrides={"nonexistent_param": 99})
        assert court is not None

    def test_mesh_height_override(self):
        court = PadelCourtModel(overrides={
            "include_mesh_above_back_walls": True,
            "back_wall_height": 4.0,
            "mesh_height": 1.5,
        })
        walls = court.get_wall_segments()
        mesh = [w for w in walls if w.surface_type == "mesh"]
        assert len(mesh) == 2
        for w in mesh:
            assert w.bounds_min[2] == pytest.approx(4.0)
            assert w.bounds_max[2] == pytest.approx(5.5)
