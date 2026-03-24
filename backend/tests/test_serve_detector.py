import pytest
from unittest.mock import MagicMock
from models.config import EventDetectorConfig
from models.types import ServerInfo, TeamId


@pytest.fixture
def config():
    return EventDetectorConfig()


@pytest.fixture
def mock_calibration():
    cal = MagicMock()
    cal.is_in_service_box.return_value = True
    return cal


@pytest.fixture
def server_near():
    return ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")


class TestServeDetection:
    def test_no_serve_without_server(self, config, mock_calibration):
        from logic.detectors.serve import ServeDetector
        sd = ServeDetector(config, mock_calibration, current_server=None)
        ball = {"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}
        result = sd.check(ball, bounce=None)
        assert result is None

    def test_valid_serve_detected(self, config, mock_calibration, server_near):
        from logic.detectors.serve import ServeDetector
        sd = ServeDetector(config, mock_calibration, current_server=server_near)
        sd.check({"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}, bounce=None)
        sd.check({"x": 5.0, "y": 5.0, "z": 1.5, "speed": 50.0}, bounce=None)
        bounce = {"court_x": 7.0, "court_y": 13.0, "side": "far", "bounce_number": 1}
        result = sd.check({"x": 7.0, "y": 13.0, "z": 0.1, "speed": 40.0}, bounce=bounce)
        assert result is not None
        assert result["valid"] is True

    def test_fault_wrong_service_box(self, config, server_near):
        from logic.detectors.serve import ServeDetector
        cal = MagicMock()
        cal.is_in_service_box.return_value = False
        sd = ServeDetector(config, cal, current_server=server_near)
        sd.check({"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}, bounce=None)
        sd.check({"x": 5.0, "y": 5.0, "z": 1.5, "speed": 50.0}, bounce=None)
        bounce = {"court_x": 3.0, "court_y": 13.0, "side": "far", "bounce_number": 1}
        result = sd.check({"x": 3.0, "y": 13.0, "z": 0.1, "speed": 40.0}, bounce=bounce)
        assert result is not None
        assert result["valid"] is False
        assert result["fault"] is True

    def test_reset_clears_state(self, config, mock_calibration, server_near):
        from logic.detectors.serve import ServeDetector
        sd = ServeDetector(config, mock_calibration, current_server=server_near)
        sd.check({"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}, bounce=None)
        sd.reset()
        assert sd._serving is False
