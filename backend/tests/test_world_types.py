"""Tests for CameraObservation, WorldState, and WallHitMetadata dataclasses."""

import pytest
from src.models.types import (
    BallPosition,
    CameraObservation,
    PlayerPosition,
    WallHitMetadata,
    WorldState,
)


class TestCameraObservation:
    def test_minimal_creation(self):
        obs = CameraObservation(camera_id="cam_left")
        assert obs.camera_id == "cam_left"
        assert obs.ball_pixel is None
        assert obs.ball_bbox is None
        assert obs.ball_court is None
        assert obs.confidence == 0.0
        assert obs.player_detections == []
        assert obs.timestamp == 0.0
        assert obs.frame_number == 0

    def test_with_ball_detection(self):
        ball_pos = BallPosition(x=1.5, y=3.0, z=0.5)
        obs = CameraObservation(
            camera_id="cam_right",
            ball_pixel=(320, 240),
            ball_bbox=[300, 220, 340, 260],
            ball_court=ball_pos,
            confidence=0.92,
            timestamp=1.234,
            frame_number=42,
        )
        assert obs.camera_id == "cam_right"
        assert obs.ball_pixel == (320, 240)
        assert obs.ball_bbox == [300, 220, 340, 260]
        assert obs.ball_court is ball_pos
        assert obs.confidence == 0.92
        assert obs.timestamp == 1.234
        assert obs.frame_number == 42

    def test_without_ball_detection(self):
        obs = CameraObservation(camera_id="cam_top", confidence=0.0)
        assert obs.ball_pixel is None
        assert obs.ball_court is None

    def test_player_detections_default_is_independent(self):
        obs1 = CameraObservation(camera_id="cam_a")
        obs2 = CameraObservation(camera_id="cam_b")
        obs1.player_detections.append({"player_id": "P1"})
        assert obs2.player_detections == []

    def test_player_detections_populated(self):
        detections = [{"player_id": "P1", "bbox": [10, 20, 50, 80]}]
        obs = CameraObservation(camera_id="cam_left", player_detections=detections)
        assert len(obs.player_detections) == 1
        assert obs.player_detections[0]["player_id"] == "P1"


class TestWorldState:
    def test_minimal_creation(self):
        ws = WorldState()
        assert ws.ball is None
        assert ws.ball_velocity is None
        assert ws.players == []
        assert ws.contributing_cameras == []
        assert ws.timestamp == 0.0
        assert ws.frame_number == 0

    def test_with_ball(self):
        ball = BallPosition(x=2.0, y=4.0, z=1.0, speed=15.0)
        ws = WorldState(
            ball=ball,
            ball_velocity=(5.0, 3.0, -1.0),
            timestamp=2.5,
            frame_number=75,
        )
        assert ws.ball is ball
        assert ws.ball_velocity == (5.0, 3.0, -1.0)
        assert ws.timestamp == 2.5
        assert ws.frame_number == 75

    def test_without_ball(self):
        ws = WorldState(timestamp=1.0, frame_number=30)
        assert ws.ball is None
        assert ws.ball_velocity is None

    def test_contributing_cameras(self):
        ws = WorldState(contributing_cameras=["cam_left", "cam_right"])
        assert len(ws.contributing_cameras) == 2
        assert "cam_left" in ws.contributing_cameras

    def test_players(self):
        players = [
            PlayerPosition(player_id="P1", x=1.0, y=2.0),
            PlayerPosition(player_id="P2", x=-1.0, y=5.0),
        ]
        ws = WorldState(players=players)
        assert len(ws.players) == 2
        assert ws.players[0].player_id == "P1"

    def test_lists_default_are_independent(self):
        ws1 = WorldState()
        ws2 = WorldState()
        ws1.players.append(PlayerPosition(player_id="P1", x=0.0, y=0.0))
        ws1.contributing_cameras.append("cam_left")
        assert ws2.players == []
        assert ws2.contributing_cameras == []


class TestWallHitMetadata:
    def test_default_creation(self):
        meta = WallHitMetadata()
        assert meta.wall_id == ""
        assert meta.surface_type == ""
        assert meta.impact_point == (0.0, 0.0, 0.0)
        assert meta.speed_at_impact == 0.0
        assert meta.incoming_angle == 0.0

    def test_populated_creation(self):
        meta = WallHitMetadata(
            wall_id="back_wall_A",
            surface_type="glass",
            impact_point=(10.0, 0.0, 1.5),
            speed_at_impact=22.5,
            incoming_angle=35.0,
        )
        assert meta.wall_id == "back_wall_A"
        assert meta.surface_type == "glass"
        assert meta.impact_point == (10.0, 0.0, 1.5)
        assert meta.speed_at_impact == 22.5
        assert meta.incoming_angle == 35.0

    def test_partial_defaults(self):
        meta = WallHitMetadata(wall_id="side_wall_B", speed_at_impact=18.0)
        assert meta.wall_id == "side_wall_B"
        assert meta.surface_type == ""
        assert meta.impact_point == (0.0, 0.0, 0.0)
        assert meta.speed_at_impact == 18.0
        assert meta.incoming_angle == 0.0
