from typing import Optional, Dict
from models.config import EventDetectorConfig
from models.types import ServerInfo, TeamId

NET_Y = 10.0
NEAR_SERVICE_ZONE_Y = (0, 4)
FAR_SERVICE_ZONE_Y = (16, 20)


class ServeDetector:
    def __init__(self, config: EventDetectorConfig, calibration,
                 current_server: Optional[ServerInfo] = None):
        self._config = config
        self._calibration = calibration
        self.current_server = current_server
        self._serving = False
        self._serve_frame_count = 0

    def check(self, ball_pos: Optional[Dict], bounce: Optional[Dict]) -> Optional[Dict]:
        if ball_pos is None or self.current_server is None:
            return None

        x, y = ball_pos["x"], ball_pos["y"]

        if not self._serving:
            if self._is_in_service_zone(y):
                self._serving = True
                self._serve_frame_count = 0
            return None

        self._serve_frame_count += 1

        if self._serve_frame_count > self._config.serve_timeout_frames:
            self._serving = False
            return {"valid": False, "fault": True, "detail": "serve_timeout"}

        if bounce is not None:
            target_side = "far" if self._is_near_side_server() else "near"
            if bounce["side"] == target_side:
                bx, by = bounce["court_x"], bounce["court_y"]
                in_box = self._calibration.is_in_service_box(bx, by, f"{target_side}_right")
                if not in_box:
                    in_box = self._calibration.is_in_service_box(bx, by, f"{target_side}_left")

                self._serving = False
                if in_box:
                    return {"valid": True, "fault": False}
                else:
                    return {"valid": False, "fault": True, "detail": "wrong_box"}
            else:
                self._serving = False
                return {"valid": False, "fault": True, "detail": "same_side"}

        return None

    def _is_near_side_server(self) -> bool:
        if self.current_server is None:
            return True
        return self.current_server.team_id == TeamId.TEAM_A

    def _is_in_service_zone(self, y: float) -> bool:
        if self._is_near_side_server():
            return NEAR_SERVICE_ZONE_Y[0] <= y <= NEAR_SERVICE_ZONE_Y[1]
        else:
            return FAR_SERVICE_ZONE_Y[0] <= y <= FAR_SERVICE_ZONE_Y[1]

    def reset(self):
        self._serving = False
        self._serve_frame_count = 0
