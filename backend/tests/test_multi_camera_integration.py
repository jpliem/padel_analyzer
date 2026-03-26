"""End-to-end test: multi-camera fusion with wall hit detection."""
import pytest

from models.court_model import PadelCourtModel
from models.types import BallPosition, CameraObservation
from pipeline.world_fusion import WorldFusion
from logic.detectors.wall_collision import WallCollisionDetector


@pytest.fixture
def court():
    return PadelCourtModel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obs(camera_id, ball_x, ball_y, ball_z, speed, confidence, timestamp, frame):
    """Build a minimal CameraObservation with a ball detection."""
    return CameraObservation(
        camera_id=camera_id,
        ball_pixel=(500, 400),
        ball_bbox=[490, 390, 510, 410],
        ball_court=BallPosition(x=ball_x, y=ball_y, z=ball_z, speed=speed),
        confidence=confidence,
        player_detections=[],
        timestamp=timestamp,
        frame_number=frame,
    )


# ---------------------------------------------------------------------------
# 1. Back near wall detected through fusion
# ---------------------------------------------------------------------------

def test_wall_hit_detected_through_fusion(court):
    """Ball detected by two cameras, trajectory crosses back near wall (Y=0)."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    # Frame 1: ball at ~(5, 2, 1.5) seen from two cameras
    obs1 = [
        _obs("cam1", 5.0, 2.0, 1.5, 70.0, 0.9, 0.033, 1),
        _obs("cam2", 5.1, 1.9, 1.4, 68.0, 0.85, 0.033, 1),
    ]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0, "cam2": 1.0})
    prev_pos = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}

    # Frame 2: ball has crossed back wall (Y < 0)
    obs2 = [
        _obs("cam1", 5.0, -0.1, 1.5, 65.0, 0.9, 0.066, 2),
    ]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0, "cam2": 1.0})
    curr_pos = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    hit = wall_detector.check(curr_pos, prev_pos)

    assert hit is not None
    assert hit["wall_id"] == "back_near"
    assert hit["surface_type"] == "glass"


# ---------------------------------------------------------------------------
# 2. No wall hit during normal rally
# ---------------------------------------------------------------------------

def test_no_wall_hit_during_normal_rally(court):
    """Ball stays in the middle of the court, no wall hit expected."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    obs1 = [_obs("cam1", 5.0, 8.0, 1.0, 50.0, 0.9, 0.033, 1)]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0})

    obs2 = [_obs("cam1", 5.0, 12.0, 0.8, 45.0, 0.9, 0.066, 2)]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0})

    prev = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}
    curr = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    hit = wall_detector.check(curr, prev)
    assert hit is None


# ---------------------------------------------------------------------------
# 3. Side wall hit through fusion
# ---------------------------------------------------------------------------

def test_side_wall_hit_through_fusion(court):
    """Ball trajectory crosses the left side wall (X=0)."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    obs1 = [_obs("cam1", 0.5, 2.0, 1.0, 40.0, 0.9, 0.033, 1)]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0})

    obs2 = [_obs("cam1", -0.2, 2.0, 1.0, 38.0, 0.9, 0.066, 2)]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0})

    prev = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}
    curr = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    hit = wall_detector.check(curr, prev)

    assert hit is not None
    assert "side_left" in hit["wall_id"]


# ---------------------------------------------------------------------------
# 4. Back far wall hit through fusion
# ---------------------------------------------------------------------------

def test_back_far_wall_hit_through_fusion(court):
    """Ball trajectory crosses the back far wall (Y=20)."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    obs1 = [_obs("cam1", 5.0, 18.5, 1.2, 55.0, 0.9, 0.033, 1)]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0})

    obs2 = [_obs("cam1", 5.0, 20.3, 1.2, 52.0, 0.9, 0.066, 2)]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0})

    prev = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}
    curr = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    hit = wall_detector.check(curr, prev)

    assert hit is not None
    assert hit["wall_id"] == "back_far"
    assert hit["surface_type"] == "glass"


# ---------------------------------------------------------------------------
# 5. Fusion weighted average from two cameras
# ---------------------------------------------------------------------------

def test_fusion_weighted_average(court):
    """WorldFusion weighted average produces position between both cameras."""
    fusion = WorldFusion(court)

    # cam1 sees ball at x=4.0, cam2 sees ball at x=6.0 — equal confidence/weight
    obs = [
        _obs("cam1", 4.0, 5.0, 1.0, 50.0, 1.0, 0.033, 1),
        _obs("cam2", 6.0, 5.0, 1.0, 50.0, 1.0, 0.033, 1),
    ]
    ws = fusion.fuse(obs, {"cam1": 1.0, "cam2": 1.0})

    assert ws.ball is not None
    assert ws.ball.x == pytest.approx(5.0, abs=0.01)
    assert ws.ball.y == pytest.approx(5.0, abs=0.01)
    assert set(ws.contributing_cameras) == {"cam1", "cam2"}


# ---------------------------------------------------------------------------
# 6. Camera with weight=0 is excluded from fusion
# ---------------------------------------------------------------------------

def test_zero_weight_camera_excluded(court):
    """Camera with weight=0 is ignored; only the other camera contributes."""
    fusion = WorldFusion(court)

    obs = [
        _obs("cam1", 3.0, 5.0, 1.0, 50.0, 0.9, 0.033, 1),
        _obs("cam2", 7.0, 5.0, 1.0, 50.0, 0.9, 0.033, 1),
    ]
    ws = fusion.fuse(obs, {"cam1": 0.0, "cam2": 1.0})

    assert ws.ball is not None
    # Only cam2 contributed — position should match cam2
    assert ws.ball.x == pytest.approx(7.0, abs=0.01)
    assert "cam1" not in ws.contributing_cameras
    assert "cam2" in ws.contributing_cameras


# ---------------------------------------------------------------------------
# 7. Single-camera observation passes through fusion unchanged
# ---------------------------------------------------------------------------

def test_single_camera_passthrough(court):
    """With one camera, fusion returns its ball_court position directly."""
    fusion = WorldFusion(court)

    obs = [_obs("cam1", 5.3, 9.7, 0.8, 60.0, 0.95, 0.033, 1)]
    ws = fusion.fuse(obs, {"cam1": 1.0})

    assert ws.ball is not None
    assert ws.ball.x == pytest.approx(5.3)
    assert ws.ball.y == pytest.approx(9.7)
    assert ws.ball.z == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 8. WorldState carries metadata from observations
# ---------------------------------------------------------------------------

def test_world_state_metadata(court):
    """WorldState timestamp and frame_number come from the first observation."""
    fusion = WorldFusion(court)

    obs = [_obs("cam1", 5.0, 10.0, 1.0, 50.0, 0.9, 1.234, 42)]
    ws = fusion.fuse(obs, {"cam1": 1.0})

    assert ws.timestamp == pytest.approx(1.234)
    assert ws.frame_number == 42


# ---------------------------------------------------------------------------
# 9. Right side wall hit through fusion
# ---------------------------------------------------------------------------

def test_right_side_wall_hit_through_fusion(court):
    """Ball trajectory crosses the right side wall (X=10)."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    obs1 = [_obs("cam1", 9.5, 2.0, 1.0, 40.0, 0.9, 0.033, 1)]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0})

    obs2 = [_obs("cam1", 10.3, 2.0, 1.0, 38.0, 0.9, 0.066, 2)]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0})

    prev = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}
    curr = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    hit = wall_detector.check(curr, prev)

    assert hit is not None
    assert "side_right" in hit["wall_id"]


# ---------------------------------------------------------------------------
# 10. Wall hit result contains required keys
# ---------------------------------------------------------------------------

def test_wall_hit_result_keys(court):
    """Wall hit dict always contains the required keys."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    obs1 = [_obs("cam1", 5.0, 1.5, 1.0, 70.0, 0.9, 0.033, 1)]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0})

    obs2 = [_obs("cam1", 5.0, -0.1, 1.0, 68.0, 0.9, 0.066, 2)]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0})

    prev = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}
    curr = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    hit = wall_detector.check(curr, prev)

    assert hit is not None
    for key in ("wall_id", "surface_type", "impact_point", "speed_at_impact", "incoming_angle"):
        assert key in hit, f"Missing key in wall hit result: {key}"
