from typing import Optional, Dict, List
from models.config import EventDetectorConfig

NET_Y = 10.0


class BounceDetector:
    def __init__(self, config: EventDetectorConfig):
        self._config = config
        self._z_history: List[float] = []
        self._speed_history: List[float] = []
        self.bounce_count: Dict[str, int] = {"near": 0, "far": 0}
        self._last_bounce_frame: int = -10

    def check(self, ball_pos: Optional[Dict]) -> Optional[Dict]:
        if ball_pos is None:
            return None

        z = ball_pos.get("z", 0.0)
        speed = ball_pos.get("speed", 0.0)
        self._z_history.append(z)
        self._speed_history.append(speed)
        if len(self._z_history) > 10:
            self._z_history = self._z_history[-10:]
        if len(self._speed_history) > 10:
            self._speed_history = self._speed_history[-10:]

        if len(self._z_history) < 3:
            return None

        descending = self._z_history[-2] > self._z_history[-1]
        was_higher = self._z_history[-3] > self._config.bounce_z_threshold
        z_low = z <= self._config.bounce_z_threshold

        recent_speeds = self._speed_history[-6:-1] if len(self._speed_history) > 5 else self._speed_history[:-1]
        if not recent_speeds:
            return None
        avg_speed = sum(recent_speeds) / len(recent_speeds)
        speed_dipped = avg_speed > 0 and speed < avg_speed * (1 - self._config.bounce_speed_dip_pct)

        if descending and was_higher and z_low and speed_dipped:
            side = "near" if ball_pos["y"] < NET_Y else "far"
            self.bounce_count[side] += 1
            return {
                "court_x": ball_pos["x"],
                "court_y": ball_pos["y"],
                "side": side,
                "bounce_number": self.bounce_count[side],
            }

        return None

    def reset(self):
        self._z_history.clear()
        self._speed_history.clear()
        self.bounce_count = {"near": 0, "far": 0}
