import pytest
from models.config import EventDetectorConfig
from models.types import PointReason


@pytest.fixture
def config():
    return EventDetectorConfig()


class TestDoubleBounce:
    def test_double_bounce_same_side(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        bounce1 = {"side": "near", "bounce_number": 1, "court_x": 3.0, "court_y": 5.0}
        result = ped.check(bounce1, ball_pos={"x": 3.0, "y": 5.0, "z": 0.1, "speed": 10.0}, ball_lost=False)
        assert result is None

        bounce2 = {"side": "near", "bounce_number": 2, "court_x": 4.0, "court_y": 6.0}
        result = ped.check(bounce2, ball_pos={"x": 4.0, "y": 6.0, "z": 0.1, "speed": 10.0}, ball_lost=False)
        assert result is not None
        assert result["reason"] == PointReason.DOUBLE_BOUNCE
        assert result["side"] == "near"


class TestBallOut:
    def test_ball_out_of_enclosure(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        ball = {"x": 12.0, "y": 5.0, "z": 0.1, "speed": 30.0}
        result = ped.check(None, ball_pos=ball, ball_lost=False)
        assert result is not None
        assert result["reason"] == PointReason.OUT

    def test_ball_inside_enclosure_but_outside_court(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        ball = {"x": 10.3, "y": 5.0, "z": 0.1, "speed": 30.0}
        result = ped.check(None, ball_pos=ball, ball_lost=False)
        assert result is None


class TestBallStopped:
    def test_ball_stopped_after_threshold(self, config):
        from logic.detectors.point_end import PointEndDetector
        config.ball_stopped_frames = 3
        ped = PointEndDetector(config)
        for _ in range(3):
            result = ped.check(None, ball_pos={"x": 5.0, "y": 5.0, "z": 0.0, "speed": 0.1}, ball_lost=False)
        assert result is not None
        assert result["reason"] == PointReason.NET


class TestBallLost:
    def test_ball_lost_triggers_point_end(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        result = ped.check(None, ball_pos=None, ball_lost=True)
        assert result is not None


class TestReset:
    def test_reset_clears_state(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        ped.check({"side": "near", "bounce_number": 1, "court_x": 3.0, "court_y": 5.0},
                  ball_pos={"x": 3.0, "y": 5.0, "z": 0.1, "speed": 10.0}, ball_lost=False)
        ped.reset()
        assert ped._bounces_per_side == {"near": 0, "far": 0}
