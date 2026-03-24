import pytest


class TestLastHitterDetector:
    def test_no_hit_on_first_frame(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        ball = {"x": 5.0, "y": 5.0, "z": 1.0, "speed": 50.0}
        players = [{"track_id": 1, "x": 5.0, "y": 5.0, "bbox": []}]
        result = lhd.check(ball, players)
        assert result is None

    def test_detects_hit_on_direction_change(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        players = [
            {"track_id": 1, "x": 3.0, "y": 3.0, "bbox": []},
            {"track_id": 2, "x": 7.0, "y": 15.0, "bbox": []},
        ]
        lhd.check({"x": 5.0, "y": 4.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 8.0, "z": 1.0, "speed": 50.0}, players)
        result = lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        assert result is not None

    def test_returns_none_when_no_ball(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        result = lhd.check(None, [])
        assert result is None

    def test_last_hitter_stored(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        players = [{"track_id": 1, "x": 5.0, "y": 5.0, "bbox": []}]
        lhd.check({"x": 5.0, "y": 4.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 8.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        assert lhd.last_hitter_track_id is not None

    def test_reset_clears(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        lhd.last_hitter_track_id = 1
        lhd.reset()
        assert lhd.last_hitter_track_id is None
