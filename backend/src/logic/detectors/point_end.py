from typing import Optional, Dict
from models.config import EventDetectorConfig
from models.types import PointReason

NET_Y = 10.0


class PointEndDetector:
    def __init__(self, config: EventDetectorConfig, court_model=None):
        self._config = config
        self._bounces_per_side: Dict[str, int] = {"near": 0, "far": 0}
        self._stopped_frames = 0
        self._frames_since_last_bounce = 0
        self._had_bounce = False
        self._last_bounce_side: Optional[str] = None
        if court_model:
            b = court_model.get_bounds()
            self._bounds = {"x_min": b["x_min"] - 0.5, "x_max": b["x_max"] + 0.5,
                            "y_min": b["y_min"] - 1.0, "y_max": b["y_max"] + 1.0}
        else:
            self._bounds = config.enclosure_bounds

    def check(self, bounce: Optional[Dict], ball_pos: Optional[Dict],
              ball_lost: bool, wall_hit=None) -> Optional[Dict]:

        # 1. Ball lost
        if ball_lost:
            return {"reason": PointReason.OUT, "detail": "ball_lost"}

        if ball_pos is None:
            return None

        x, y = ball_pos.get("x", 0), ball_pos.get("y", 0)

        # 2. Ball out of enclosure
        if (x < self._bounds["x_min"] or x > self._bounds["x_max"] or
                y < self._bounds["y_min"] or y > self._bounds["y_max"]):
            return {"reason": PointReason.OUT, "x": x, "y": y}

        # 3. Wall before bounce (from WallCollisionDetector)
        if wall_hit and self._had_bounce:
            ball_side = "near" if y < NET_Y else "far"
            if self._bounces_per_side[ball_side] == 0:
                return {"reason": PointReason.WALL_BEFORE_BOUNCE, "side": ball_side}

        # 4. Ball stopped
        speed = ball_pos.get("speed", 0)
        if speed < 1.0:
            self._stopped_frames += 1
        else:
            self._stopped_frames = 0

        if self._stopped_frames >= self._config.ball_stopped_frames:
            return {"reason": PointReason.NET, "detail": "ball_stopped"}

        # 5. Double bounce
        if bounce is not None:
            side = bounce["side"]
            self._bounces_per_side[side] += 1
            self._had_bounce = True
            self._last_bounce_side = side
            self._frames_since_last_bounce = 0
            if self._bounces_per_side[side] >= 2:
                return {"reason": PointReason.DOUBLE_BOUNCE, "side": side}
            other = "far" if side == "near" else "near"
            self._bounces_per_side[other] = 0
        else:
            self._frames_since_last_bounce += 1

        # 6. Winner timeout
        if (self._had_bounce and
                self._frames_since_last_bounce >= self._config.winner_timeout_frames):
            return {"reason": PointReason.WINNER, "side": self._last_bounce_side}

        return None

    def reset(self):
        self._bounces_per_side = {"near": 0, "far": 0}
        self._stopped_frames = 0
        self._frames_since_last_bounce = 0
        self._had_bounce = False
        self._last_bounce_side = None
