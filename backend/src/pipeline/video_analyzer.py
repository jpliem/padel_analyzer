from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import numpy as np
import cv2

from models.config import EventDetectorConfig
from models.types import MatchEvent, MatchConfig, ServerInfo, TeamId
from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector, YoloPlayerDetector
from cv.detectors.tracknet import TrackNetBallDetector
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
                 match_config: MatchConfig = None,
                 detector_type: str = "yolo",
                 tracknet_model_path: str = "models/tracknet_tennis.pt"):
        config = config or EventDetectorConfig()
        match_config = match_config or MatchConfig()

        # Always create YOLO (needed for player detection + ball fallback)
        unified = UnifiedYoloDetector()
        self.player_detector = YoloPlayerDetector(unified)

        # Ball detector: YOLO or TrackNet
        if detector_type == "tracknet":
            yolo_fallback = YoloBallDetector(unified)
            self.ball_detector = TrackNetBallDetector(
                model_path=tracknet_model_path,
                yolo_fallback=yolo_fallback,
            )
        else:
            self.ball_detector = YoloBallDetector(unified)

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
        self.player_positions_log: List[Dict] = []  # per-frame player positions

    def process_frame(self, frame: np.ndarray, frame_no: int) -> FrameResult:
        self._frame_count = frame_no
        ball_bbox = self.ball_detector.detect(frame, frame_no)
        player_detections = self.player_detector.detect(frame, frame_no)
        ball_pos = self.ball_tracker.update(ball_bbox, frame_no)
        player_pos = self.player_tracker.update(player_detections, frame_no)

        # Re-assign players every 30 frames to handle track ID changes
        if frame_no >= self._config.auto_assign_after_frames and frame_no % 30 == 0:
            self._auto_assign_players(player_pos)
            self._auto_assigned = True

        events = self.event_detector.process(ball_pos, player_pos, frame_no)
        self.all_events.extend(events)

        # Log player positions per frame (lightweight: just court coords)
        self.player_positions_log.append({
            "frame": frame_no,
            "players": [
                {"track_id": p["track_id"], "x": p["x"], "y": p["y"],
                 "player_id": self.player_tracker.get_player_id(p["track_id"])}
                for p in player_pos
            ],
        })

        return FrameResult(
            ball_position=ball_pos,
            player_positions=player_pos,
            events=events,
            score=self.scoring_engine.get_score_display(),
            frame_number=frame_no,
        )

    def analyze_video(self, video_path: str,
                      progress_callback: Optional[Callable] = None,
                      annotated_path: Optional[str] = None) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ball_tracker.fps = fps
        self.ball_tracker.dt = 1.0 / fps

        writer = None
        if annotated_path:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))

        frame_no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = self.process_frame(frame, frame_no)

            if writer:
                annotated = self._draw_overlays(frame, result)
                writer.write(annotated)

            frame_no += 1

            if progress_callback and frame_no % 30 == 0:
                pct = (frame_no / total_frames * 100) if total_frames > 0 else 0
                progress_callback(frame_no, total_frames, pct)

        cap.release()
        if writer:
            writer.release()

        return {
            "match_id": self._match_id,
            "frames_processed": frame_no,
            "events": len(self.all_events),
            "final_score": self.scoring_engine.get_score_display(),
        }

    def _draw_overlays(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        """Draw detection bounding boxes, ball trail, and score on the frame."""
        out = frame.copy()

        # Draw player bounding boxes
        for p in result.player_positions:
            bbox = p.get("bbox", [])
            if len(bbox) == 4:
                x1, y1, x2, y2 = [int(v) for v in bbox]
                # Determine team color
                player_id = self.player_tracker.get_player_id(p["track_id"])
                if player_id and player_id in ("P1", "P2"):
                    color = (255, 185, 116)  # blue (BGR)
                else:
                    color = (85, 112, 225)  # red (BGR)
                cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
                label = player_id or f"#{p['track_id']}"
                cv2.putText(out, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw ball position
        if result.ball_position and result.ball_position.get("detected"):
            # Get ball pixel position from trajectory (reverse of court coords)
            # Use raw detection bbox from detector cache
            pass

        # Draw ball trail on court (last 15 trajectory points as dots)
        trail = self.ball_tracker.trajectory[-15:]
        for i, tp in enumerate(trail):
            try:
                px, py = self.ball_tracker.calibration.court_to_pixel(tp["x"], tp["y"])
                alpha = 0.3 + 0.7 * (i / max(len(trail) - 1, 1))
                radius = max(2, int(4 * alpha))
                cv2.circle(out, (int(px), int(py)), radius, (0, 230, 255), -1)
            except Exception:
                pass

        # Draw ball current position (larger, brighter)
        if result.ball_position:
            try:
                px, py = self.ball_tracker.calibration.court_to_pixel(
                    result.ball_position["x"], result.ball_position["y"])
                cv2.circle(out, (int(px), int(py)), 8, (0, 255, 255), -1)
                cv2.circle(out, (int(px), int(py)), 10, (0, 200, 255), 2)
                # Speed label
                speed = result.ball_position.get("speed", 0)
                if speed > 0:
                    cv2.putText(out, f"{speed:.0f} km/h", (int(px) + 12, int(py) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            except Exception:
                pass

        # Draw score overlay (top center)
        score = result.score
        score_text = f"{score['score']}  |  G: {score['games']}  |  S: {score['sets']}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        tx = (out.shape[1] - text_size[0]) // 2
        # Background rectangle
        cv2.rectangle(out, (tx - 10, 5), (tx + text_size[0] + 10, 35), (0, 0, 0), -1)
        cv2.putText(out, score_text, (tx, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return out

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
