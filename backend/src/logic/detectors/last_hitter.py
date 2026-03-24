import numpy as np
from typing import Optional, Dict, List


class LastHitterDetector:
    def __init__(self, angle_threshold_deg: float = 90.0):
        self._angle_threshold = np.radians(angle_threshold_deg)
        self._prev_positions: List[Dict] = []
        self.last_hitter_track_id: Optional[int] = None

    def check(self, ball_pos: Optional[Dict], player_positions: List[Dict]) -> Optional[Dict]:
        if ball_pos is None:
            return None

        self._prev_positions.append({"x": ball_pos["x"], "y": ball_pos["y"]})
        if len(self._prev_positions) > 5:
            self._prev_positions = self._prev_positions[-5:]
        if len(self._prev_positions) < 3:
            return None

        p1 = self._prev_positions[-3]
        p2 = self._prev_positions[-2]
        p3 = self._prev_positions[-1]

        v1 = np.array([p2["x"] - p1["x"], p2["y"] - p1["y"]])
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 0.01 or norm2 < 0.01:
            return None

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle >= self._angle_threshold:
            rev_x, rev_y = p2["x"], p2["y"]
            closest_id = self._find_closest(rev_x, rev_y, player_positions)
            if closest_id is not None:
                self.last_hitter_track_id = closest_id
                return {"track_id": closest_id, "x": rev_x, "y": rev_y}

        return None

    def _find_closest(self, x: float, y: float, players: List[Dict]) -> Optional[int]:
        min_dist = float("inf")
        closest = None
        for p in players:
            dist = np.sqrt((p["x"] - x) ** 2 + (p["y"] - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = p["track_id"]
        return closest

    def reset(self):
        self._prev_positions.clear()
        self.last_hitter_track_id = None
