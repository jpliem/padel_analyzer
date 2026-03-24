import pytest
from models.config import EventDetectorConfig


@pytest.fixture
def config():
    return EventDetectorConfig()


class TestBounceDetection:
    def test_no_bounce_when_ball_high(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        ball = {"x": 5.0, "y": 5.0, "z": 2.0, "speed": 50.0}
        result = bd.check(ball)
        assert result is None

    def test_bounce_detected_on_z_drop(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        for z in [3.0, 2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 5.0, "z": z, "speed": 60.0})
        result = bd.check({"x": 5.0, "y": 5.0, "z": 0.1, "speed": 25.0})
        assert result is not None
        assert result["court_x"] == 5.0
        assert result["court_y"] == 5.0

    def test_bounce_records_court_side(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        for z in [2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 15.0, "z": z, "speed": 60.0})
        result = bd.check({"x": 5.0, "y": 15.0, "z": 0.1, "speed": 25.0})
        assert result is not None
        assert result["side"] == "far"

    def test_bounce_count_increments(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        for z in [2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 5.0, "z": z, "speed": 60.0})
        bd.check({"x": 5.0, "y": 5.0, "z": 0.1, "speed": 25.0})
        assert bd.bounce_count["near"] == 1
        assert bd.bounce_count["far"] == 0

    def test_no_bounce_when_none_ball(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        result = bd.check(None)
        assert result is None

    def test_reset_clears_state(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        for z in [2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 5.0, "z": z, "speed": 60.0})
        bd.check({"x": 5.0, "y": 5.0, "z": 0.1, "speed": 25.0})
        bd.reset()
        assert bd.bounce_count["near"] == 0
        assert bd.bounce_count["far"] == 0
