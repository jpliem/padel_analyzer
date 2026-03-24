from typing import Optional, Dict, List
from models.config import EventDetectorConfig
from models.types import MatchState, MatchEvent, EventType, CourtPoint, PointReason, TeamId
from logic.match_state_machine import MatchStateMachine
from logic.detectors.bounce import BounceDetector
from logic.detectors.last_hitter import LastHitterDetector
from logic.detectors.serve import ServeDetector
from logic.detectors.point_end import PointEndDetector


class EventDetector:
    def __init__(self, config: EventDetectorConfig, calibration,
                 scoring_engine, player_tracker, team_map: Dict[str, int]):
        self._config = config
        self._calibration = calibration
        self._scoring_engine = scoring_engine
        self._player_tracker = player_tracker
        self._team_map = team_map

        self.state_machine = MatchStateMachine()
        self._bounce_detector = BounceDetector(config)
        self._last_hitter = LastHitterDetector()
        self._serve_detector = ServeDetector(
            config, calibration,
            current_server=getattr(scoring_engine, 'current_server', None)
        )
        self._point_end_detector = PointEndDetector(config)

    def process(self, ball_pos: Optional[Dict], player_positions: List[Dict],
                frame_no: int) -> List[MatchEvent]:
        events: List[MatchEvent] = []

        self._serve_detector.current_server = getattr(
            self._scoring_engine, 'current_server', None
        )

        bounce = self._bounce_detector.check(ball_pos)
        hit = self._last_hitter.check(ball_pos, player_positions)
        ball_lost = ball_pos is None and hasattr(self, '_had_ball') and self._had_ball

        if ball_pos is not None:
            self._had_ball = True

        state = self.state_machine.state

        if state == MatchState.IDLE:
            if ball_pos is not None and self._serve_detector.current_server is not None:
                serve_check = self._serve_detector.check(ball_pos, bounce)
                if self._serve_detector._serving:
                    self.state_machine.on_serve_started()

        elif state in (MatchState.SERVING_1ST, MatchState.SERVING_2ND):
            if ball_pos is not None:
                serve_result = self._serve_detector.check(ball_pos, bounce)
                if serve_result is not None:
                    self.state_machine.on_serve_result(serve_result["valid"])
                    if serve_result.get("fault"):
                        events.append(self._make_event(EventType.FAULT, frame_no, ball_pos))

                    if self.state_machine.state == MatchState.POINT_ENDED:
                        self._resolve_point_end(PointReason.DOUBLE_FAULT, None, last_hitter_track_id=None)
                        events.append(self._make_event(EventType.POINT_END, frame_no, ball_pos, {"reason": "double_fault"}))
                        self.state_machine.on_score_updated()
                        self._reset_detectors()
                    elif serve_result["valid"]:
                        events.append(self._make_event(EventType.SERVE, frame_no, ball_pos))

        elif state == MatchState.RALLY:
            if bounce is not None:
                events.append(self._make_event(EventType.BOUNCE, frame_no, ball_pos, {"side": bounce["side"]}))

            if hit is not None:
                events.append(self._make_event(EventType.HIT, frame_no, ball_pos, {"track_id": hit["track_id"]}))

            point_end = self._point_end_detector.check(bounce, ball_pos, ball_lost)
            if point_end is not None:
                reason = point_end["reason"]
                side = point_end.get("side")
                self.state_machine.on_point_ended(reason)
                self._resolve_point_end(reason, side, self._last_hitter.last_hitter_track_id)
                events.append(self._make_event(EventType.POINT_END, frame_no, ball_pos, {"reason": reason.value}))
                self.state_machine.on_score_updated()
                self._reset_detectors()

        return events

    def _resolve_point_end(self, reason: PointReason, side: Optional[str],
                           last_hitter_track_id: Optional[int]):
        winner_team = self._determine_winner(reason, side, last_hitter_track_id)
        if winner_team is not None:
            self._scoring_engine.add_point(winner_team, reason)

    def _determine_winner(self, reason: PointReason, side: Optional[str],
                          last_hitter_track_id: Optional[int]) -> Optional[int]:
        if reason == PointReason.DOUBLE_FAULT:
            server = getattr(self._scoring_engine, 'current_server', None)
            if server:
                return 2 if server.team_id == TeamId.TEAM_A else 1
            return 2

        if reason == PointReason.DOUBLE_BOUNCE:
            if side == "near":
                return 2
            return 1

        if reason in (PointReason.OUT, PointReason.NET, PointReason.WALL_BEFORE_BOUNCE):
            if last_hitter_track_id is not None:
                player_id = self._player_tracker.get_player_id(last_hitter_track_id)
                if player_id and player_id in self._team_map:
                    hitter_team = self._team_map[player_id]
                    return 2 if hitter_team == 1 else 1
            return None

        if reason == PointReason.WINNER:
            if last_hitter_track_id is not None:
                player_id = self._player_tracker.get_player_id(last_hitter_track_id)
                if player_id and player_id in self._team_map:
                    return self._team_map[player_id]
            return None

        return None

    def _reset_detectors(self):
        self._bounce_detector.reset()
        self._last_hitter.reset()
        self._serve_detector.reset()
        self._point_end_detector.reset()

    @staticmethod
    def _make_event(event_type: EventType, frame_no: int,
                    ball_pos: Optional[Dict], metadata: Dict = None) -> MatchEvent:
        x = ball_pos["x"] if ball_pos else 0.0
        y = ball_pos["y"] if ball_pos else 0.0
        return MatchEvent(
            event_type=event_type,
            timestamp=ball_pos.get("timestamp", 0.0) if ball_pos else 0.0,
            frame_number=frame_no,
            position=CourtPoint(x=x, y=y),
            metadata=metadata or {},
        )
