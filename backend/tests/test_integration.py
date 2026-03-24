"""
Integration tests for the full pipeline: calibration → ball tracker → player tracker → scoring engine.
Tests that all components work together as a system.
"""
import numpy as np
import pytest
from cv.court_calibration import CourtCalibration
from cv.ball_tracker import BallTracker
from cv.player_tracker import PlayerTracker
from logic.scoring_engine import PadelScoringEngine
from models.types import ServerInfo, TeamId


@pytest.fixture
def calibrated_court(sample_court_corners_pixels):
    """Fixture: fully calibrated court ready for tracking."""
    cal = CourtCalibration()
    cal.calibrate(sample_court_corners_pixels)
    return cal


class TestPipelineIntegration:
    def test_calibration_feeds_ball_tracker(self, calibrated_court):
        """Test that a calibrated court feeds into BallTracker correctly."""
        tracker = BallTracker(calibrated_court, fps=30)
        pos = tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert pos is not None
        assert "x" in pos and "y" in pos

    def test_calibration_feeds_player_tracker(self, calibrated_court):
        """Test that a calibrated court feeds into PlayerTracker correctly."""
        tracker = PlayerTracker(calibrated_court)
        dets = np.array([[100, 200, 160, 400, 0.9, 0]])
        positions = tracker.update(dets, frame_number=0)
        assert len(positions) == 1

    def test_full_pipeline_point_flow(self, calibrated_court):
        """Simulate a full point: detection → tracking → scoring.

        Flow:
        1. Ball moves through 3 frames, tracked at each step
        2. Players detected and assigned IDs
        3. Point awarded to team 1
        4. Verify score updates and trajectory recorded
        """
        ball_tracker = BallTracker(calibrated_court, fps=30)
        player_tracker = PlayerTracker(calibrated_court)
        engine = PadelScoringEngine(
            first_server=ServerInfo(team_id=TeamId.TEAM_A, player_id="P1"),
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]},
        )

        # Simulate ball moving across court
        ball_positions = [
            [500, 400, 520, 420],
            [600, 420, 620, 440],
            [700, 450, 720, 470],
        ]
        for i, bbox in enumerate(ball_positions):
            pos = ball_tracker.update(bbox=bbox, frame_number=i)
            assert pos is not None

        # Detect and assign players
        player_dets = np.array([
            [100, 200, 160, 400, 0.9, 0],
            [400, 200, 460, 400, 0.85, 0],
        ])
        player_tracker.update(player_dets, frame_number=0)
        player_tracker.assign_player(track_id=1, player_id="P1")
        player_tracker.assign_player(track_id=2, player_id="P2")

        # Award point and verify score
        engine.add_point(1)
        score = engine.get_score_display()
        assert score["score"] == "15 - 0"

        # Verify trajectory and speed recorded
        assert len(ball_tracker.trajectory) == 3
        assert ball_tracker.trajectory[-1]["speed"] > 0

    def test_scoring_engine_with_full_config(self):
        """Test scoring engine with full configuration and multi-point gameplay.

        Flow:
        1. Configure engine with golden_point=True, sets_to_win=1
        2. Play enough points to win: 6 games * 4 points each = 24 points
        3. Verify game and set counted correctly
        """
        engine = PadelScoringEngine(
            golden_point=True,
            sets_to_win=1,
            first_server=ServerInfo(team_id=TeamId.TEAM_A, player_id="P1"),
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]},
        )

        # Play 6 games worth of points (6 * 4 = 24 points to team 1)
        for _ in range(6):
            for _ in range(4):
                engine.add_point(1)

        assert engine.game_over is True
        assert engine.team1_sets == 1
