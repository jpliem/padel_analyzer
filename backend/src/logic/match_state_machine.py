from typing import Optional
from models.types import MatchState, PointReason


class MatchStateMachine:
    def __init__(self):
        self.state = MatchState.IDLE
        self.point_end_reason: Optional[PointReason] = None

    def on_serve_started(self):
        if self.state == MatchState.IDLE:
            self.state = MatchState.SERVING_1ST

    def on_serve_result(self, valid: bool):
        if self.state == MatchState.SERVING_1ST:
            if valid:
                self.state = MatchState.RALLY
            else:
                self.state = MatchState.SERVING_2ND
        elif self.state == MatchState.SERVING_2ND:
            if valid:
                self.state = MatchState.RALLY
            else:
                self.point_end_reason = PointReason.DOUBLE_FAULT
                self.state = MatchState.POINT_ENDED

    def on_let(self):
        pass

    def on_point_ended(self, reason: PointReason):
        if self.state == MatchState.RALLY:
            self.point_end_reason = reason
            self.state = MatchState.POINT_ENDED

    def on_score_updated(self):
        if self.state == MatchState.POINT_ENDED:
            self.state = MatchState.IDLE
            self.point_end_reason = None

    def reset_to_idle(self):
        self.state = MatchState.IDLE
        self.point_end_reason = None
