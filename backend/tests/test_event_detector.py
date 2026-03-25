import pytest
from unittest.mock import MagicMock
from models.config import EventDetectorConfig
from models.types import MatchState, PointReason, TeamId, ServerInfo, EventType


@pytest.fixture
def config():
    return EventDetectorConfig()


@pytest.fixture
def mock_calibration():
    cal = MagicMock()
    cal.is_in_service_box.return_value = True
    cal.is_in_bounds.return_value = True
    return cal


@pytest.fixture
def mock_scoring():
    eng = MagicMock()
    eng.current_server = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
    eng.get_score_display.return_value = {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"}
    return eng


@pytest.fixture
def mock_player_tracker():
    pt = MagicMock()
    pt.find_closest_player.return_value = "P1"
    pt.get_player_id.return_value = "P1"
    return pt


@pytest.fixture
def team_map():
    return {"P1": 1, "P2": 1, "P3": 2, "P4": 2}


class TestEventDetector:
    def test_init_state_idle(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        assert ed.state_machine.state == MatchState.IDLE

    def test_process_returns_list(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        ball = {"x": 5.0, "y": 5.0, "z": 2.0, "speed": 50.0}
        players = [{"track_id": 1, "x": 5.0, "y": 5.0, "bbox": []}]
        events = ed.process(ball, players, frame_no=0)
        assert isinstance(events, list)

    def test_process_handles_none_ball(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        events = ed.process(None, [], frame_no=0)
        assert isinstance(events, list)

    def test_point_end_calls_add_point(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        ed.state_machine.state = MatchState.RALLY
        ed.state_machine.on_point_ended(PointReason.DOUBLE_BOUNCE)
        ed._resolve_point_end(PointReason.DOUBLE_BOUNCE, "near", last_hitter_track_id=1)
        mock_scoring.add_point.assert_called_once()


def test_wall_hit_event_emitted(mock_calibration, mock_scoring, mock_player_tracker):
    from logic.event_detector import EventDetector
    from models.court_model import PadelCourtModel
    config = EventDetectorConfig()
    court = PadelCourtModel()
    team_map = {"P1": 1, "P2": 1, "P3": 2, "P4": 2}
    ed = EventDetector(config, mock_calibration, mock_scoring,
                       mock_player_tracker, team_map, court_model=court)
    # Put into RALLY state
    ed.state_machine.on_serve_started()
    ed.state_machine.on_serve_result(True)
    # First frame: ball in court
    ed.process(ball_pos={"x": 5.0, "y": 1.0, "z": 1.5, "speed": 60.0},
               player_positions=[], frame_no=99)
    # Second frame: ball crossed back wall
    events = ed.process(
        ball_pos={"x": 5.0, "y": -0.1, "z": 1.5, "speed": 60.0},
        player_positions=[], frame_no=100)
    wall_events = [e for e in events if e.event_type == EventType.WALL_HIT]
    assert len(wall_events) >= 1
    assert "wall_id" in wall_events[0].metadata
