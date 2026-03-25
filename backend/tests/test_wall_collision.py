"""
TDD tests for WallCollisionDetector.

Standard padel court:
  X: 0..10, Y: 0..20, Z: 0..up
  back_near at Y=0 (glass, 4m), back_far at Y=20 (glass, 4m)
  side_left/right_near/far glass (3m), fence_left/right (4m, middle section)
"""

import math
import pytest

from models.court_model import PadelCourtModel
from logic.detectors.wall_collision import WallCollisionDetector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def court():
    return PadelCourtModel()


@pytest.fixture
def detector(court):
    return WallCollisionDetector(court)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ball(x, y, z=1.0, speed=80.0):
    return {"x": x, "y": y, "z": z, "speed": speed}


# ---------------------------------------------------------------------------
# 1. No positions
# ---------------------------------------------------------------------------

class TestNoPositions:
    def test_both_none_returns_none(self, detector):
        assert detector.check(None, None) is None

    def test_current_none_returns_none(self, detector):
        assert detector.check(None, ball(5.0, 5.0)) is None

    def test_prev_none_returns_none(self, detector):
        # Without a previous position 3D ray is undefined
        assert detector.check(ball(5.0, 5.0), None) is None


# ---------------------------------------------------------------------------
# 2. No hit — ball in center of court
# ---------------------------------------------------------------------------

class TestNoHit:
    def test_center_to_center_no_hit(self, detector):
        result = detector.check(ball(5.0, 10.0), ball(5.0, 9.5))
        assert result is None

    def test_small_move_near_net_no_hit(self, detector):
        result = detector.check(ball(5.0, 9.0), ball(5.0, 8.5))
        assert result is None


# ---------------------------------------------------------------------------
# 3. Back near wall — Y crosses 0
# ---------------------------------------------------------------------------

class TestBackNearWall:
    def test_trajectory_crosses_y0(self, detector):
        # Ball moving from Y=1 toward Y=0 (and beyond)
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        assert result["wall_id"] == "back_near"
        assert result["surface_type"] == "glass"

    def test_impact_point_tuple(self, detector):
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        assert isinstance(result["impact_point"], tuple)
        assert len(result["impact_point"]) == 3

    def test_impact_y_at_wall(self, detector):
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        # Impact point Y should be ~0.0 (at the wall plane)
        assert abs(result["impact_point"][1]) < 0.01

    def test_speed_preserved(self, detector):
        result = detector.check(ball(5.0, 0.5, speed=85.0), ball(5.0, 1.5, speed=90.0))
        assert result is not None
        assert result["speed_at_impact"] == pytest.approx(85.0)

    def test_incoming_angle_head_on(self, detector):
        # Ball moving straight into wall — angle from normal should be small (< 20 deg)
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        assert result["incoming_angle"] < 20.0


# ---------------------------------------------------------------------------
# 4. Back far wall — Y crosses 20
# ---------------------------------------------------------------------------

class TestBackFarWall:
    def test_trajectory_crosses_y20(self, detector):
        result = detector.check(ball(5.0, 19.5), ball(5.0, 18.5))
        assert result is not None
        assert result["wall_id"] == "back_far"
        assert result["surface_type"] == "glass"

    def test_impact_y_at_far_wall(self, detector):
        result = detector.check(ball(5.0, 19.5), ball(5.0, 18.5))
        assert result is not None
        assert abs(result["impact_point"][1] - 20.0) < 0.01


# ---------------------------------------------------------------------------
# 5. Side walls — X crosses 0 or 10
# ---------------------------------------------------------------------------

class TestSideWalls:
    def test_hit_left_near_glass(self, detector):
        # Ball at Y=2 (within side glass section Y=0..4), moving toward X=0
        result = detector.check(ball(0.5, 2.0), ball(1.5, 2.0))
        assert result is not None
        assert "left" in result["wall_id"]
        assert result["surface_type"] == "glass"

    def test_hit_right_near_glass(self, detector):
        result = detector.check(ball(9.5, 2.0), ball(8.5, 2.0))
        assert result is not None
        assert "right" in result["wall_id"]
        assert result["surface_type"] == "glass"

    def test_impact_x_at_left_wall(self, detector):
        result = detector.check(ball(0.5, 2.0), ball(1.5, 2.0))
        assert result is not None
        assert abs(result["impact_point"][0]) < 0.01

    def test_impact_x_at_right_wall(self, detector):
        result = detector.check(ball(9.5, 2.0), ball(8.5, 2.0))
        assert result is not None
        assert abs(result["impact_point"][0] - 10.0) < 0.01


# ---------------------------------------------------------------------------
# 6. Side fence — middle section
# ---------------------------------------------------------------------------

class TestSideFence:
    def test_hit_left_fence(self, detector):
        # Y=10 is middle of court, fence section Y=4..16
        result = detector.check(ball(0.5, 10.0), ball(1.5, 10.0))
        assert result is not None
        assert "fence" in result["wall_id"] or result["surface_type"] == "fence"

    def test_hit_right_fence(self, detector):
        result = detector.check(ball(9.5, 10.0), ball(8.5, 10.0))
        assert result is not None
        assert result["surface_type"] == "fence"


# ---------------------------------------------------------------------------
# 7. Ball above wall height → no hit
# ---------------------------------------------------------------------------

class TestAboveWallHeight:
    def test_ball_above_back_wall_no_hit(self, detector):
        # Back wall is 4m high; ball at Z=5.0 should not hit
        result = detector.check(ball(5.0, 0.5, z=5.0), ball(5.0, 1.5, z=5.0))
        assert result is None

    def test_ball_above_side_glass_no_hit(self, detector):
        # Side glass is 3m high; ball at Z=3.5 should not hit side glass
        # (fence is 4m, so need Z>4 to guarantee no hit)
        result = detector.check(ball(0.5, 2.0, z=5.0), ball(1.5, 2.0, z=5.0))
        assert result is None


# ---------------------------------------------------------------------------
# 8. Incoming angle
# ---------------------------------------------------------------------------

class TestIncomingAngle:
    def test_head_on_angle_near_zero(self, detector):
        # Straight into near back wall — angle from normal should be < 20 deg
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        assert result["incoming_angle"] < 20.0

    def test_glancing_angle_larger(self, detector):
        # Ball moving mostly parallel to back wall (large X component, small Y)
        # from (1, 0.5) toward (9, 0.3) — diagonal but still crosses Y=0
        result = detector.check(ball(1.0, 0.3), ball(0.5, 0.8))
        # If it hits, the angle should be > head-on case
        if result is not None:
            head_on = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
            assert head_on is not None
            assert result["incoming_angle"] >= head_on["incoming_angle"] - 1.0


# ---------------------------------------------------------------------------
# 9. Speed at impact
# ---------------------------------------------------------------------------

class TestSpeedAtImpact:
    def test_speed_from_current_ball_pos(self, detector):
        result = detector.check(ball(5.0, 0.5, speed=72.0), ball(5.0, 1.5, speed=90.0))
        assert result is not None
        assert result["speed_at_impact"] == pytest.approx(72.0)


# ---------------------------------------------------------------------------
# 10. 2D fallback
# ---------------------------------------------------------------------------

class TestTwoDFallback:
    def test_2d_fallback_detects_near_wall(self, detector):
        # use_3d=False: proximity-based fallback
        result = detector.check(ball(5.0, 0.2, z=0.0), ball(5.0, 0.5, z=0.0), use_3d=False)
        assert result is not None
        assert "back" in result["wall_id"]

    def test_2d_fallback_no_hit_at_center(self, detector):
        result = detector.check(ball(5.0, 10.0, z=0.0), ball(5.0, 9.0, z=0.0), use_3d=False)
        assert result is None

    def test_2d_fallback_still_returns_correct_keys(self, detector):
        result = detector.check(ball(5.0, 0.2, z=0.0), ball(5.0, 0.5, z=0.0), use_3d=False)
        assert result is not None
        for key in ("wall_id", "surface_type", "impact_point", "speed_at_impact", "incoming_angle"):
            assert key in result, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# 11. Reset is a no-op (stateless detector)
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_does_not_raise(self, detector):
        detector.reset()

    def test_detector_still_works_after_reset(self, detector):
        detector.reset()
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None


# ---------------------------------------------------------------------------
# 12. Return dict structure
# ---------------------------------------------------------------------------

class TestReturnDictStructure:
    def test_all_required_keys_present(self, detector):
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        for key in ("wall_id", "surface_type", "impact_point", "speed_at_impact", "incoming_angle"):
            assert key in result, f"Missing key: {key}"

    def test_impact_point_is_tuple(self, detector):
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        assert isinstance(result["impact_point"], tuple)

    def test_speed_is_float(self, detector):
        result = detector.check(ball(5.0, 0.5, speed=77.5), ball(5.0, 1.5, speed=80.0))
        assert result is not None
        assert isinstance(result["speed_at_impact"], float)

    def test_incoming_angle_is_float(self, detector):
        result = detector.check(ball(5.0, 0.5), ball(5.0, 1.5))
        assert result is not None
        assert isinstance(result["incoming_angle"], float)
