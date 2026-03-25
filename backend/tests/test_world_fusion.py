"""Tests for WorldFusion — multi-camera observation merging."""

import pytest
from models.types import BallPosition, CameraObservation, PlayerPosition, WorldState
from models.court_model import PadelCourtModel
from pipeline.world_fusion import WorldFusion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(camera_id, x, y, z=0.0, speed=0.0, conf=0.8,
              pixel=(500, 300), frame=1, ts=0.033):
    return CameraObservation(
        camera_id=camera_id,
        ball_pixel=pixel,
        ball_bbox=[490, 290, 510, 310],
        ball_court=BallPosition(x=x, y=y, z=z, speed=speed, timestamp=ts),
        confidence=conf,
        player_detections=[],
        timestamp=ts,
        frame_number=frame,
    )


def _make_obs_no_ball(camera_id, ts=0.033, frame=1):
    return CameraObservation(
        camera_id=camera_id,
        ball_pixel=None,
        ball_bbox=None,
        ball_court=None,
        confidence=0.0,
        player_detections=[],
        timestamp=ts,
        frame_number=frame,
    )


def _make_obs_with_players(camera_id, players, x=5.0, y=10.0, ts=0.033, frame=1):
    """Observation with player detections. players is a list of dicts with player_id, x, y."""
    return CameraObservation(
        camera_id=camera_id,
        ball_pixel=(500, 300),
        ball_bbox=[490, 290, 510, 310],
        ball_court=BallPosition(x=x, y=y, z=0.0, speed=0.0, timestamp=ts),
        confidence=0.8,
        player_detections=players,
        timestamp=ts,
        frame_number=frame,
    )


@pytest.fixture
def court():
    return PadelCourtModel()


@pytest.fixture
def fusion(court):
    return WorldFusion(court)


# ---------------------------------------------------------------------------
# Ball fusion tests
# ---------------------------------------------------------------------------

class TestSingleCameraFusion:
    def test_passthrough_ball_position(self, fusion):
        obs = _make_obs("cam_left", x=3.0, y=7.0, z=1.5, conf=0.9)
        result = fusion.fuse([obs], {"cam_left": 1.0})

        assert result.ball is not None
        assert result.ball.x == pytest.approx(3.0)
        assert result.ball.y == pytest.approx(7.0)
        assert result.ball.z == pytest.approx(1.5)

    def test_contributing_camera_recorded(self, fusion):
        obs = _make_obs("cam_left", x=3.0, y=7.0)
        result = fusion.fuse([obs], {"cam_left": 1.0})

        assert "cam_left" in result.contributing_cameras

    def test_timestamp_and_frame_propagated(self, fusion):
        obs = _make_obs("cam_left", x=3.0, y=7.0, ts=1.234, frame=37)
        result = fusion.fuse([obs], {"cam_left": 1.0})

        assert result.timestamp == pytest.approx(1.234)
        assert result.frame_number == 37


class TestTwoCameraWeightedAverage:
    def test_equal_weights_midpoint(self, fusion):
        obs_a = _make_obs("cam_left", x=4.0, y=10.0, conf=0.8)
        obs_b = _make_obs("cam_right", x=6.0, y=10.0, conf=0.8)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        assert result.ball is not None
        assert result.ball.x == pytest.approx(5.0, abs=1e-6)

    def test_unequal_weights_bias(self, fusion):
        obs_a = _make_obs("cam_left", x=4.0, y=10.0, conf=0.8)
        obs_b = _make_obs("cam_right", x=6.0, y=10.0, conf=0.8)

        # cam_left has 3x the weight, so result should be closer to 4.0
        result = fusion.fuse([obs_a, obs_b], {"cam_left": 3.0, "cam_right": 1.0})

        assert result.ball is not None
        assert result.ball.x == pytest.approx(4.5, abs=1e-6)

    def test_unequal_confidence_bias(self, fusion):
        obs_a = _make_obs("cam_left", x=4.0, y=10.0, conf=0.9)
        obs_b = _make_obs("cam_right", x=6.0, y=10.0, conf=0.1)

        # cam_left has higher confidence, result closer to 4.0
        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        assert result.ball is not None
        assert result.ball.x < 5.0

    def test_both_cameras_in_contributing(self, fusion):
        obs_a = _make_obs("cam_left", x=4.0, y=10.0)
        obs_b = _make_obs("cam_right", x=6.0, y=10.0)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        assert "cam_left" in result.contributing_cameras
        assert "cam_right" in result.contributing_cameras

    def test_z_coordinate_averaged(self, fusion):
        obs_a = _make_obs("cam_left", x=5.0, y=10.0, z=2.0, conf=0.8)
        obs_b = _make_obs("cam_right", x=5.0, y=10.0, z=4.0, conf=0.8)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        assert result.ball.z == pytest.approx(3.0, abs=1e-6)


class TestNoBallObservations:
    def test_no_cameras_first_frame_returns_none(self, fusion):
        result = fusion.fuse([], {})
        assert result.ball is None

    def test_no_ball_detections_returns_none_first_frame(self, fusion):
        obs = _make_obs_no_ball("cam_left")
        result = fusion.fuse([obs], {"cam_left": 1.0})
        assert result.ball is None

    def test_no_observations_returns_prev_ball(self, fusion):
        # First frame: establish a ball position
        obs = _make_obs("cam_left", x=5.0, y=10.0, ts=0.033, frame=1)
        fusion.fuse([obs], {"cam_left": 1.0})

        # Second frame: no observations — should return prev_ball as prediction
        result = fusion.fuse([], {})
        assert result.ball is not None
        assert result.ball.x == pytest.approx(5.0)
        assert result.ball.y == pytest.approx(10.0)

    def test_zero_weight_camera_excluded(self, fusion):
        obs_a = _make_obs("cam_left", x=5.0, y=10.0, conf=0.8)
        obs_b = _make_obs("cam_right", x=9.0, y=10.0, conf=0.8)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 0.0})

        assert result.ball is not None
        assert result.ball.x == pytest.approx(5.0, abs=1e-6)
        assert "cam_right" not in result.contributing_cameras


class TestVelocityComputation:
    def test_velocity_none_on_first_frame(self, fusion):
        obs = _make_obs("cam_left", x=5.0, y=10.0, ts=0.033)
        result = fusion.fuse([obs], {"cam_left": 1.0})
        assert result.ball_velocity is None

    def test_velocity_computed_on_second_frame(self, fusion):
        obs1 = _make_obs("cam_left", x=5.0, y=10.0, ts=0.0)
        fusion.fuse([obs1], {"cam_left": 1.0})

        obs2 = _make_obs("cam_left", x=6.0, y=10.0, ts=1.0, frame=2)
        result = fusion.fuse([obs2], {"cam_left": 1.0})

        assert result.ball_velocity is not None
        vx, vy, vz = result.ball_velocity
        assert vx == pytest.approx(1.0, abs=1e-4)  # moved +1m in 1s
        assert vy == pytest.approx(0.0, abs=1e-4)

    def test_velocity_sign_correct(self, fusion):
        obs1 = _make_obs("cam_left", x=8.0, y=15.0, ts=0.0)
        fusion.fuse([obs1], {"cam_left": 1.0})

        obs2 = _make_obs("cam_left", x=6.0, y=12.0, ts=1.0, frame=2)
        result = fusion.fuse([obs2], {"cam_left": 1.0})

        vx, vy, vz = result.ball_velocity
        assert vx < 0  # x decreased
        assert vy < 0  # y decreased

    def test_velocity_reset_after_reset(self, fusion):
        obs1 = _make_obs("cam_left", x=5.0, y=10.0, ts=0.0)
        fusion.fuse([obs1], {"cam_left": 1.0})

        fusion.reset()

        obs2 = _make_obs("cam_left", x=6.0, y=10.0, ts=1.0, frame=2)
        result = fusion.fuse([obs2], {"cam_left": 1.0})
        assert result.ball_velocity is None


# ---------------------------------------------------------------------------
# Player deduplication tests
# ---------------------------------------------------------------------------

class TestPlayerDeduplication:
    def test_single_camera_players_passthrough(self, fusion):
        players = [
            {"player_id": "P1", "court_x": 2.0, "court_y": 5.0, "confidence": 0.9},
            {"player_id": "P2", "court_x": 8.0, "court_y": 5.0, "confidence": 0.9},
        ]
        obs = _make_obs_with_players("cam_left", players)
        result = fusion.fuse([obs], {"cam_left": 1.0})

        assert len(result.players) == 2

    def test_two_cameras_same_player_deduped(self, fusion):
        """Two cameras see the same player at slightly different positions → 1 merged player."""
        players_a = [{"player_id": "P1", "court_x": 2.0, "court_y": 5.0, "confidence": 0.9}]
        players_b = [{"player_id": "P1", "court_x": 2.1, "court_y": 5.05, "confidence": 0.8}]

        obs_a = _make_obs_with_players("cam_left", players_a)
        obs_b = _make_obs_with_players("cam_right", players_b)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        assert len(result.players) == 1
        # Position should be weighted average
        assert result.players[0].x == pytest.approx(
            (2.0 * 0.9 + 2.1 * 0.8) / (0.9 + 0.8), abs=1e-4
        )

    def test_two_cameras_different_players_not_deduped(self, fusion):
        """Two cameras see players far apart → 2 distinct players."""
        players_a = [{"player_id": "P1", "court_x": 2.0, "court_y": 3.0, "confidence": 0.9}]
        players_b = [{"player_id": "P2", "court_x": 8.0, "court_y": 17.0, "confidence": 0.9}]

        obs_a = _make_obs_with_players("cam_left", players_a)
        obs_b = _make_obs_with_players("cam_right", players_b)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        assert len(result.players) == 2

    def test_dedup_threshold_boundary(self, fusion):
        """Players exactly at PLAYER_DEDUP_DISTANCE should NOT be merged."""
        from pipeline.world_fusion import PLAYER_DEDUP_DISTANCE

        players_a = [{"player_id": "P1", "court_x": 0.0, "court_y": 0.0, "confidence": 0.9}]
        # Place second detection exactly at the dedup distance
        players_b = [{"player_id": "P1", "court_x": PLAYER_DEDUP_DISTANCE, "court_y": 0.0, "confidence": 0.9}]

        obs_a = _make_obs_with_players("cam_left", players_a)
        obs_b = _make_obs_with_players("cam_right", players_b)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        # At exactly the threshold, should not be merged (< threshold to merge)
        assert len(result.players) == 2

    def test_dedup_just_inside_threshold(self, fusion):
        """Players just inside PLAYER_DEDUP_DISTANCE should be merged."""
        from pipeline.world_fusion import PLAYER_DEDUP_DISTANCE

        players_a = [{"player_id": "P1", "court_x": 0.0, "court_y": 0.0, "confidence": 0.9}]
        players_b = [{"player_id": "P1", "court_x": PLAYER_DEDUP_DISTANCE - 0.01, "court_y": 0.0, "confidence": 0.9}]

        obs_a = _make_obs_with_players("cam_left", players_a)
        obs_b = _make_obs_with_players("cam_right", players_b)

        result = fusion.fuse([obs_a, obs_b], {"cam_left": 1.0, "cam_right": 1.0})

        assert len(result.players) == 1

    def test_no_player_detections(self, fusion):
        obs = _make_obs("cam_left", x=5.0, y=10.0)
        result = fusion.fuse([obs], {"cam_left": 1.0})
        assert result.players == []


# ---------------------------------------------------------------------------
# Reset tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_prev_ball(self, fusion):
        obs = _make_obs("cam_left", x=5.0, y=10.0, ts=0.033)
        fusion.fuse([obs], {"cam_left": 1.0})
        fusion.reset()

        # After reset, no observations should return None (not prev ball)
        result = fusion.fuse([], {})
        assert result.ball is None

    def test_reset_clears_prev_time(self, fusion):
        obs1 = _make_obs("cam_left", x=5.0, y=10.0, ts=0.0)
        fusion.fuse([obs1], {"cam_left": 1.0})
        fusion.reset()

        obs2 = _make_obs("cam_left", x=6.0, y=10.0, ts=1.0)
        result = fusion.fuse([obs2], {"cam_left": 1.0})
        # No velocity since prev_time was reset
        assert result.ball_velocity is None
