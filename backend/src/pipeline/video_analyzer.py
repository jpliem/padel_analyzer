from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import numpy as np
import cv2

from models.config import EventDetectorConfig
from models.types import MatchEvent, MatchConfig, ServerInfo, TeamId
from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector, YoloPlayerDetector
from cv.detectors.tracknet import TrackNetBallDetector
from cv.detectors.fast_ball import FastBallDetector
from cv.ball_tracker import BallTracker
from cv.player_tracker import PlayerTracker
from cv.court_detector import CourtDetector
from cv.court_calibration import CourtCalibration
from logic.event_detector import EventDetector
from logic.scoring_engine import PadelScoringEngine
from models.court_model import PadelCourtModel
from pipeline.world_fusion import WorldFusion
from cv.visibility import BallVisibilityTracker
from logic.padel_rules import PadelRulesEngine
from models.observations import ObservationKind, PadelObservation, RuleDecision
from logic.review_ledger import ReviewLedger
from models.types import PointReason
from cv.monocular_trajectory import MonocularTrajectoryEstimator, RayObservation
from cv.active_ball import ActiveBallSelector
from cv.model_registry import get_ball_model_profile
from cv.audio_events import extract_video_audio_evidence
from logic.contact_fusion import ContactEvidence, fuse_contact
from logic.semantic_bridge import SemanticEventBridge
from models.observations import DecisionKind


@dataclass
class FrameResult:
    ball_position: Optional[Dict]
    player_positions: List[Dict]
    events: List[MatchEvent]
    score: Dict
    frame_number: int
    ball_visibility: Dict = field(default_factory=dict)
    rule_decisions: List[RuleDecision] = field(default_factory=list)
    monocular_trajectory: Optional[Dict] = None
    active_ball: Dict = field(default_factory=dict)


class VideoAnalyzer:
    def __init__(self, match_id: str, calibration,
                 config: EventDetectorConfig = None,
                 match_config: MatchConfig = None,
                 detector_type: str = "yolo",
                 tracknet_model_path: Optional[str] = None,
                 court_model_overrides: Dict = None,
                 pose_model_path: Optional[str] = None):
        config = config or EventDetectorConfig()
        match_config = match_config or MatchConfig()

        # Always create YOLO (needed for player detection + ball fallback)
        unified = UnifiedYoloDetector()
        self.player_detector = YoloPlayerDetector(unified)
        self.pose_detector = None
        if pose_model_path:
            from cv.player_pose import PlayerPoseDetector
            self.pose_detector = PlayerPoseDetector(pose_model_path)

        # Ball detector: YOLO, TrackNet, or Fast (frame differencing)
        self.model_info = get_ball_model_profile(tracknet_model_path)
        if detector_type == "tracknet":
            yolo_fallback = YoloBallDetector(unified)
            self.ball_detector = TrackNetBallDetector(
                model_path=self.model_info["checkpoint_path"],
                yolo_fallback=yolo_fallback,
            )
        elif detector_type == "fast":
            yolo_fallback = YoloBallDetector(unified)
            self.ball_detector = FastBallDetector(yolo_fallback=yolo_fallback)
        else:
            self.ball_detector = YoloBallDetector(unified)
            self.model_info = {
                "id": f"{detector_type}-ball-detector",
                "task": "single_camera_ball_detection",
                "status": "unbenchmarked_runtime_option",
                "evidence": None,
                "limitations": ["No reviewed-label accuracy result is registered for this runtime option."],
            }

        self.ball_tracker = BallTracker(calibration)
        self.active_ball_selector = ActiveBallSelector()
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

        # Court model and world fusion for multi-camera support
        self._court_model = PadelCourtModel(overrides=court_model_overrides)
        self._world_fusion = WorldFusion(self._court_model)
        self._visibility_tracker = BallVisibilityTracker()
        self._monocular_estimator = MonocularTrajectoryEstimator()
        self._ray_observations: List[RayObservation] = []
        self._audio_evidence = {
            "status": "not_analyzed", "events": [],
            "warning": "Audio evidence is prepared when a recording is analyzed.",
        }
        self._audio_by_frame: Dict[int, float] = {}
        self._pixel_motion: List[tuple] = []
        self.contact_proposals: List[Dict] = []
        self.rules_engine = PadelRulesEngine(
            out_of_court_play_enabled=match_config.out_of_court_play_enabled,
        )
        self.semantic_bridge = SemanticEventBridge(team_map)
        self.semantic_observations: List[Dict] = []
        self.rule_decisions_log: List[Dict] = []
        self.review_ledger = ReviewLedger(
            golden_point=match_config.golden_point,
            sets_to_win=match_config.format.value if match_config.format else 2,
            first_server=first_server,
            team_players=team_players,
        )

        self.event_detector = EventDetector(
            config, calibration, self.scoring_engine,
            self.player_tracker, team_map,
            court_model=self._court_model,
            review_ledger=self.review_ledger,
        )

        self._config = config
        self._calibration = calibration
        self._match_id = match_id
        self._auto_assigned = False
        self._frame_count = 0
        self.cancel_requested = False
        self.all_events: List[MatchEvent] = []
        self.player_positions_log: List[Dict] = []

        # Auto court detection — update calibration per frame
        import os
        court_model_path = "models/court_keypoints.pt"
        if os.path.exists(court_model_path):
            try:
                self._court_detector = CourtDetector(model_path=court_model_path)
                self._auto_court = True
            except Exception:
                self._court_detector = None
                self._auto_court = False
        else:
            self._court_detector = None
            self._auto_court = False

    def _update_calibration(self, frame: np.ndarray) -> None:
        """Auto-detect court keypoints and update homography."""
        if not self._auto_court or self._court_detector is None:
            return
        keypoints = self._court_detector.detect(frame)
        if keypoints and len(keypoints) == 12:
            try:
                cal = CourtCalibration()
                cal.calibrate_keypoints(keypoints)
                # Update all components that use calibration
                self.ball_tracker.calibration = cal
                self.player_tracker.calibration = cal
                self._calibration = cal
            except Exception:
                pass  # Keep previous calibration on error

    def process_frame(self, frame: np.ndarray, frame_no: int) -> FrameResult:
        self._frame_count = frame_no

        # Update calibration from auto-detected keypoints (every 5 frames to save compute)
        if frame_no % 5 == 0:
            self._update_calibration(frame)

        if hasattr(self.ball_detector, "detect_candidates"):
            ball_candidates = self.ball_detector.detect_candidates(frame, frame_no)
        else:
            detected_bbox = self.ball_detector.detect(frame, frame_no)
            ball_candidates = ([{
                "bbox": detected_bbox, "confidence": 0.5,
                "source": self.ball_detector.__class__.__name__,
            }] if detected_bbox is not None else [])
        active_selection = self.active_ball_selector.select(ball_candidates, frame.shape)
        ball_bbox = active_selection.bbox
        player_detections = self.player_detector.detect(frame, frame_no)
        player_poses = self.pose_detector.detect(frame) if self.pose_detector else []
        ball_pos = self.ball_tracker.update(ball_bbox, frame_no)
        player_pos = self.player_tracker.update(player_detections, frame_no, frame=frame)

        monocular_fit = None
        if ball_bbox is not None and hasattr(self._calibration, "pixel_ray"):
            try:
                px = (ball_bbox[0] + ball_bbox[2]) / 2.0
                py = (ball_bbox[1] + ball_bbox[3]) / 2.0
                origin, direction = self._calibration.pixel_ray(px, py)
                timestamp = frame_no * self.ball_tracker.dt
                self._ray_observations.append(RayObservation(
                    timestamp=timestamp, camera_origin=origin,
                    ray_direction=direction, frame_number=frame_no,
                ))
                self._ray_observations = self._ray_observations[-12:]
                fit = self._monocular_estimator.fit(self._ray_observations)
                if fit is not None:
                    monocular_fit = {
                        "reliable": fit.reliable, "confidence": fit.confidence,
                        "median_ray_error_m": fit.median_ray_error_m,
                        "condition_number": fit.condition_number,
                    }
                    if fit.reliable and ball_pos is not None:
                        point = fit.points[-1]
                        ball_pos.update({
                            "x": point.x, "y": point.y, "z": max(0.0, point.z),
                            "position_source": "monocular_ballistic_fit",
                            "position_confidence": fit.confidence,
                        })
            except (RuntimeError, ValueError, np.linalg.LinAlgError):
                pass

        direction_confidence = self._direction_change_confidence(ball_bbox)
        audio_confidence = self._audio_by_frame.get(frame_no, 0.0)
        if audio_confidence > 0 or direction_confidence >= 0.45:
            proposal = fuse_contact(ContactEvidence(
                frame_number=frame_no,
                audio_confidence=audio_confidence,
                direction_change_confidence=direction_confidence,
            ))
            # This evidence is diagnostic/review material. It never mutates score.
            self.contact_proposals.append({
                "frame_number": frame_no,
                "timestamp": frame_no * self.ball_tracker.dt,
                "contact_type": proposal.contact_type,
                "confidence": proposal.confidence,
                "requires_review": proposal.requires_review,
                "evidence": list(proposal.evidence),
            })

        # A detector miss is recorded as uncertainty.  It is never translated
        # directly into OUT or a point award.
        recoverable = False
        if ball_bbox is None and ball_pos is not None:
            recoverable = self._court_model.is_recoverable_position(
                ball_pos["x"], ball_pos["y"]
            )
        visibility = self._visibility_tracker.update(
            detected=ball_bbox is not None,
            outside_court_recoverable=recoverable,
        )
        visibility_observation = PadelObservation(
            kind=ObservationKind.VISIBILITY_CHANGED,
            timestamp=(ball_pos or {}).get("timestamp", 0.0),
            frame_number=frame_no,
            confidence=visibility.confidence,
            visibility=visibility.state,
            position=((ball_pos["x"], ball_pos["y"], ball_pos.get("z", 0.0))
                      if ball_pos else None),
            metadata={"missing_frames": visibility.missing_frames},
        )
        visibility_decision = self.rules_engine.process(visibility_observation)

        # Initial assignment once, then only assign new unmatched tracks
        if not self._auto_assigned and frame_no >= self._config.auto_assign_after_frames:
            self._auto_assign_players(player_pos)
            self._auto_assigned = True
        elif self._auto_assigned:
            self._assign_new_tracks(player_pos)

        events = self.event_detector.process(ball_pos, player_pos, frame_no)
        semantic_decisions = []
        for event in events:
            if event.event_type.value == "HIT":
                track_id = event.metadata.get("track_id")
                event.metadata["player_id"] = self.player_tracker.get_player_id(track_id)
            server = getattr(self.scoring_engine, "current_server", None)
            server_team = getattr(server, "team_id", None)
            event_confidence = max(0.0, min(1.0, active_selection.confidence * 0.85))
            observations = self.semantic_bridge.translate(event, server_team, event_confidence)
            for observation in observations:
                decision = self.rules_engine.process(observation)
                semantic_decisions.append(decision)
                self.semantic_observations.append(self._observation_json(observation))
                self.rule_decisions_log.append(self._decision_json(decision, frame_no))
                if decision.kind == DecisionKind.SERVE_STARTED:
                    self.semantic_bridge.point_ready_accepted()
                if decision.kind == DecisionKind.POINT_AWARDED and decision.winner is not None:
                    # Semantic CV decisions remain human-confirmed until their
                    # perception path has a separate scoring-accuracy benchmark.
                    self.review_ledger.propose(
                        frame_number=frame_no,
                        winner_team=decision.winner.value,
                        reason=self._point_reason(decision.reason),
                        confidence=decision.confidence,
                        source="semantic_rules",
                        auto_confirm_threshold=2.0,
                    )
                    self.semantic_bridge.point_ended()
        # EventDetector replays the immutable point ledger after every award.
        self.scoring_engine = self.event_detector._scoring_engine
        self.all_events.extend(events)

        # Log player positions per frame (lightweight: just court coords).
        # Drop degenerate projections: (0,0) means a failed transform, and
        # positions far outside the court are noise; a padel court holds 4.
        valid_players = [
            p for p in player_pos
            if not (p["x"] == 0 and p["y"] == 0)
            and -5 <= p["x"] <= 15 and -5 <= p["y"] <= 25
        ][:4]
        self.player_positions_log.append({
            "frame": frame_no,
            "players": [
                {"track_id": p["track_id"], "x": p["x"], "y": p["y"],
                 "player_id": self.player_tracker.get_player_id(p["track_id"])}
                for p in valid_players
            ],
            "poses": [
                {"bbox": pose.bbox, "keypoints": pose.keypoints,
                 "confidence": pose.confidence}
                for pose in player_poses
            ],
        })

        return FrameResult(
            ball_position=ball_pos,
            player_positions=player_pos,
            events=events,
            score=self.scoring_engine.get_score_display(),
            frame_number=frame_no,
            ball_visibility={
                "state": visibility.state.value,
                "missing_frames": visibility.missing_frames,
                "confidence": visibility.confidence,
            },
            rule_decisions=[visibility_decision, *semantic_decisions],
            monocular_trajectory=monocular_fit,
            active_ball=active_selection.as_dict(),
        )

    def add_manual_point(self, team: int, frame_number: Optional[int] = None):
        record = self.review_ledger.propose(
            frame_number=self._frame_count if frame_number is None else frame_number,
            winner_team=team,
            reason=PointReason.MANUAL,
            confidence=1.0,
            source="manual",
        )
        self.scoring_engine = self.review_ledger.replay()
        self.event_detector._scoring_engine = self.scoring_engine
        return record

    def refresh_score_from_reviews(self):
        self.scoring_engine = self.review_ledger.replay()
        self.event_detector._scoring_engine = self.scoring_engine
        return self.scoring_engine.get_score_display()

    def analyze_video(self, video_path: str,
                      progress_callback: Optional[Callable] = None,
                      annotated_path: Optional[str] = None,
                      max_frames: Optional[int] = None) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None and max_frames > 0:
            total_frames = min(total_frames, max_frames) if total_frames > 0 else max_frames
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.ball_tracker.fps = fps
        self.ball_tracker.dt = 1.0 / fps
        self._audio_evidence = extract_video_audio_evidence(video_path)
        self._audio_by_frame = {}
        for event in self._audio_evidence.get("events", []):
            frame = int(round(float(event["timestamp"]) * fps))
            confidence = float(event.get("confidence", 0.0))
            # Give fusion a one-frame tolerance for A/V alignment.
            for offset in (-1, 0, 1):
                target = max(0, frame + offset)
                self._audio_by_frame[target] = max(
                    confidence, self._audio_by_frame.get(target, 0.0))

        writer = None
        if annotated_path:
            # Try H.264 (avc1) first; many opencv-python builds (esp. macOS) lack
            # the H.264 encoder and open() fails silently, producing a 0-byte file.
            # Fall back to mp4v, which ships with every opencv build.
            for codec in ("avc1", "mp4v"):
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(annotated_path, fourcc, fps, (w, h))
                if writer.isOpened():
                    break
                writer.release()
                writer = None

        frame_no = 0
        cancelled = False
        while cap.isOpened():
            if self.cancel_requested:
                cancelled = True
                break
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_no >= max_frames:
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
            "cancelled": cancelled,
            "events": len(self.all_events),
            "final_score": self.scoring_engine.get_score_display(),
            "model_info": self.model_info,
            "active_ball_diagnostics": self.active_ball_selector.diagnostics(),
            "evidence_status": {
                "audio": {key: value for key, value in self._audio_evidence.items()
                          if key != "events"},
                "audio_impulses": len(self._audio_evidence.get("events", [])),
                "pose": "enabled" if self.pose_detector else "not_configured",
                "contact_proposals": len(self.contact_proposals),
                "semantic_rule_decisions": len(self.rule_decisions_log),
                "scoring_policy": "Evidence creates review material and never awards a point alone.",
            },
            "contact_proposals": self.contact_proposals,
            "semantic_observations": self.semantic_observations,
            "rule_decisions": self.rule_decisions_log,
        }

    @staticmethod
    def _observation_json(observation: PadelObservation) -> Dict:
        return {
            "kind": observation.kind.value, "timestamp": observation.timestamp,
            "frame_number": observation.frame_number,
            "confidence": observation.confidence,
            "team": observation.team.value if observation.team else None,
            "player_id": observation.player_id, "side": observation.side,
            "surface": observation.surface.value if observation.surface else None,
            "position": observation.position, "metadata": observation.metadata,
        }

    @staticmethod
    def _decision_json(decision: RuleDecision, frame_number: int) -> Dict:
        return {
            "frame_number": frame_number, "kind": decision.kind.value,
            "phase": decision.phase.value, "reason": decision.reason,
            "winner": decision.winner.value if decision.winner else None,
            "confidence": decision.confidence,
            "requires_confirmation": decision.requires_confirmation,
            "evidence": list(decision.evidence),
        }

    @staticmethod
    def _point_reason(reason: str) -> PointReason:
        if reason.startswith("double_fault"):
            return PointReason.DOUBLE_FAULT
        if "second_bounce" in reason:
            return PointReason.DOUBLE_BOUNCE
        if "wall" in reason or "fence" in reason:
            return PointReason.WALL_BEFORE_BOUNCE
        if "net" in reason:
            return PointReason.NET
        if "exit" in reason or "outside" in reason:
            return PointReason.OUT
        return PointReason.WINNER

    def _direction_change_confidence(self, bbox: Optional[List[float]]) -> float:
        if bbox is None:
            return 0.0
        center = ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)
        self._pixel_motion.append(center)
        self._pixel_motion = self._pixel_motion[-3:]
        if len(self._pixel_motion) < 3:
            return 0.0
        p0, p1, p2 = self._pixel_motion
        v1 = np.array([p1[0] - p0[0], p1[1] - p0[1]], dtype=float)
        v2 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
        n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if min(n1, n2) < 2.0:
            return 0.0
        cosine = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        angle_score = float(np.arccos(cosine) / np.pi)
        motion_score = min(1.0, min(n1, n2) / 8.0)
        return angle_score * motion_score

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

        self._last_known_positions = {}
        for track_id, player_id in assignments:
            self.player_tracker.assign_player(track_id, player_id)
            # Store last known position for each player ID
            pos = next((p for p in current_positions if p["track_id"] == track_id), None)
            if pos:
                self._last_known_positions[player_id] = (pos["x"], pos["y"])

    def _assign_new_tracks(self, current_positions: List[Dict]):
        """Assign player IDs to new track IDs by proximity to last known position."""
        if not hasattr(self, '_last_known_positions'):
            return

        for p in current_positions:
            tid = p["track_id"]
            existing = self.player_tracker.get_player_id(tid)
            if existing:
                # Track already has a player ID — update last known position
                self._last_known_positions[existing] = (p["x"], p["y"])
                continue

            # New unassigned track — find the closest unassigned player ID
            assigned_pids = set()
            for pp in current_positions:
                pid = self.player_tracker.get_player_id(pp["track_id"])
                if pid:
                    assigned_pids.add(pid)

            # Find unassigned player IDs
            all_pids = {"P1", "P2", "P3", "P4"}
            unassigned = all_pids - assigned_pids

            if not unassigned:
                continue

            # Match to closest last known position
            best_pid = None
            best_dist = float("inf")
            for pid in unassigned:
                if pid in self._last_known_positions:
                    lx, ly = self._last_known_positions[pid]
                    dist = ((p["x"] - lx) ** 2 + (p["y"] - ly) ** 2) ** 0.5
                    if dist < best_dist:
                        best_dist = dist
                        best_pid = pid

            if best_pid and best_dist < 5.0:  # max 5 meters to match
                self.player_tracker.assign_player(tid, best_pid)
                self._last_known_positions[best_pid] = (p["x"], p["y"])
