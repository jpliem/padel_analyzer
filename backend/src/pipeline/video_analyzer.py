from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import numpy as np
import cv2

from models.config import EventDetectorConfig
from models.types import MatchEvent, MatchConfig, ServerInfo, TeamId
from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector, YoloPlayerDetector
from cv.ball_tracker import BallTracker
from cv.player_tracker import PlayerTracker
from logic.event_detector import EventDetector
from logic.scoring_engine import PadelScoringEngine


@dataclass
class FrameResult:
    ball_position: Optional[Dict]
    player_positions: List[Dict]
    events: List[MatchEvent]
    score: Dict
    frame_number: int


class VideoAnalyzer:
    def __init__(self, match_id: str, calibration,
                 config: EventDetectorConfig = None,
                 match_config: MatchConfig = None):
        config = config or EventDetectorConfig()
        match_config = match_config or MatchConfig()

        unified = UnifiedYoloDetector()
        self.ball_detector = YoloBallDetector(unified)
        self.player_detector = YoloPlayerDetector(unified)

        self.ball_tracker = BallTracker(calibration)
        self.player_tracker = PlayerTracker(calibration)

        team_players = match_config.teams if match_config.teams else None
        first_server = match_config.first_server if match_config.first_server else None
        self.scoring_engine = PadelScoringEngine(
            golden_point=match_config.golden_point,
            sets_to_win=match_config.format.value if match_config.format else 2,
            first_server=first_server,
            team_players=team_players,
        )

        team_map = {}
        if match_config.teams:
            for team_id, players in match_config.teams.items():
                tid = team_id.value if isinstance(team_id, TeamId) else team_id
                for pid in players:
                    team_map[pid] = tid
        if not team_map:
            team_map = {"P1": 1, "P2": 1, "P3": 2, "P4": 2}

        self.event_detector = EventDetector(
            config, calibration, self.scoring_engine,
            self.player_tracker, team_map
        )

        self._config = config
        self._match_id = match_id
        self._auto_assigned = False
        self._frame_count = 0
        self.all_events: List[MatchEvent] = []

    def process_frame(self, frame: np.ndarray, frame_no: int) -> FrameResult:
        self._frame_count = frame_no
        ball_bbox = self.ball_detector.detect(frame, frame_no)
        player_detections = self.player_detector.detect(frame, frame_no)
        ball_pos = self.ball_tracker.update(ball_bbox, frame_no)
        player_pos = self.player_tracker.update(player_detections, frame_no)

        if not self._auto_assigned and frame_no >= self._config.auto_assign_after_frames:
            self._auto_assign_players(player_pos)
            self._auto_assigned = True

        events = self.event_detector.process(ball_pos, player_pos, frame_no)
        self.all_events.extend(events)

        return FrameResult(
            ball_position=ball_pos,
            player_positions=player_pos,
            events=events,
            score=self.scoring_engine.get_score_display(),
            frame_number=frame_no,
        )

    def analyze_video(self, video_path: str,
                      progress_callback: Optional[Callable] = None) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ball_tracker.fps = fps
        self.ball_tracker.dt = 1.0 / fps

        frame_no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, frame_no)
            frame_no += 1

            if progress_callback and frame_no % 30 == 0:
                pct = (frame_no / total_frames * 100) if total_frames > 0 else 0
                progress_callback(frame_no, total_frames, pct)

        cap.release()
        return {
            "match_id": self._match_id,
            "frames_processed": frame_no,
            "events": len(self.all_events),
            "final_score": self.scoring_engine.get_score_display(),
        }

    def _auto_assign_players(self, current_positions: List[Dict]):
        if len(current_positions) < 2:
            return

        near = sorted([p for p in current_positions if p["y"] < 10.0],
                      key=lambda p: p["x"])
        far = sorted([p for p in current_positions if p["y"] >= 10.0],
                     key=lambda p: p["x"])

        assignments = []
        for i, p in enumerate(near[:2]):
            pid = f"P{i + 1}"
            assignments.append((p["track_id"], pid))
        for i, p in enumerate(far[:2]):
            pid = f"P{i + 3}"
            assignments.append((p["track_id"], pid))

        for track_id, player_id in assignments:
            self.player_tracker.assign_player(track_id, player_id)
