from models.types import (
    TeamId, PointReason, MatchFormat, ServerInfo,
    MatchConfig, CourtPoint, BallPosition, PlayerPosition,
    EventType, MatchEvent
)


def test_team_id_values():
    assert TeamId.TEAM_A.value == 1
    assert TeamId.TEAM_B.value == 2


def test_point_reason_values():
    assert PointReason.WINNER.value == "winner"
    assert PointReason.DOUBLE_FAULT.value == "double_fault"
    assert PointReason.OUT.value == "out"
    assert PointReason.NET.value == "net"
    assert PointReason.DOUBLE_BOUNCE.value == "double_bounce"


def test_match_format_values():
    assert MatchFormat.BEST_OF_3.value == 2
    assert MatchFormat.BEST_OF_1.value == 1


def test_server_info():
    s = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
    assert s.team_id == TeamId.TEAM_A
    assert s.player_id == "P1"


def test_match_config_defaults():
    cfg = MatchConfig(
        match_name="Test Match",
        players={"P1": "Alice", "P2": "Bob", "P3": "Carol", "P4": "Dave"},
        teams={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]},
    )
    assert cfg.golden_point is True
    assert cfg.format == MatchFormat.BEST_OF_3
    assert cfg.first_server == ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")


def test_court_point():
    p = CourtPoint(x=5.0, y=10.0)
    assert p.x == 5.0
    assert p.y == 10.0


def test_ball_position():
    bp = BallPosition(x=5.0, y=10.0, z=1.2, speed=45.0, timestamp=1.5)
    assert bp.z == 1.2
    assert bp.speed == 45.0


def test_event_type_values():
    assert EventType.BOUNCE.value == "BOUNCE"
    assert EventType.SERVE.value == "SERVE"
    assert EventType.FAULT.value == "FAULT"
    assert EventType.LET.value == "LET"
    assert EventType.NET_HIT.value == "NET_HIT"
    assert EventType.WALL_HIT.value == "WALL_HIT"
    assert EventType.OUT.value == "OUT"
    assert EventType.POINT_END.value == "POINT_END"


def test_match_event():
    evt = MatchEvent(
        event_type=EventType.BOUNCE,
        timestamp=12.5,
        frame_number=375,
        position=CourtPoint(x=3.0, y=7.0),
        confidence=0.85,
    )
    assert evt.event_type == EventType.BOUNCE
    assert evt.position.x == 3.0
    assert evt.metadata == {}


def test_match_state_values():
    from models.types import MatchState
    assert MatchState.IDLE.value == "IDLE"
    assert MatchState.SERVING_1ST.value == "SERVING_1ST"
    assert MatchState.SERVING_2ND.value == "SERVING_2ND"
    assert MatchState.RALLY.value == "RALLY"
    assert MatchState.POINT_ENDED.value == "POINT_ENDED"
    assert MatchState.SCORE_UPDATE.value == "SCORE_UPDATE"


def test_event_type_hit():
    from models.types import EventType
    assert EventType.HIT.value == "HIT"


def test_event_detector_config_defaults():
    from models.config import EventDetectorConfig
    cfg = EventDetectorConfig()
    assert cfg.bounce_z_threshold == 0.3
    assert cfg.bounce_speed_dip_pct == 0.4
    assert cfg.serve_timeout_frames == 90
    assert cfg.winner_timeout_frames == 60
    assert cfg.ball_stopped_frames == 15
    assert cfg.auto_assign_after_frames == 30


def test_event_detector_config_custom():
    from models.config import EventDetectorConfig
    cfg = EventDetectorConfig(bounce_z_threshold=0.5, winner_timeout_frames=90)
    assert cfg.bounce_z_threshold == 0.5
    assert cfg.winner_timeout_frames == 90
