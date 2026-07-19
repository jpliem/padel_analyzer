import numpy as np
import supervision as sv
from typing import List, Dict, Optional
from cv.player_reid import PlayerAppearanceEncoder, PlayerReIdentifier


class PlayerTracker:
    """Player tracker using supervision's ByteTrack for stable track IDs."""

    # Enclosure bounds for filtering non-players
    COURT_X_MIN, COURT_X_MAX = -2.0, 12.0
    COURT_Y_MIN, COURT_Y_MAX = -3.0, 23.0

    def __init__(self, calibration, fps: float = 30.0):
        """Accept either CourtCalibration or CameraModel for projection."""
        self.calibration = calibration
        self._byte_track = sv.ByteTrack(
            frame_rate=int(fps),
            track_activation_threshold=0.4,
            lost_track_buffer=60,
            minimum_matching_threshold=0.7,
            minimum_consecutive_frames=1,
        )
        self._tracks: Dict[int, Dict] = {}
        self._player_map: Dict[int, str] = {}  # track_id → player_id (P1-P4)
        try:
            from cv.player_reid import OsnetAppearanceEncoder
            self._appearance_encoder = OsnetAppearanceEncoder()
            # ReID-embedding cosine similarities sit lower than HSV histogram
            # overlaps; matched pairs typically score 0.55-0.8.
            self._reid = PlayerReIdentifier(similarity_threshold=0.60,
                                            margin_threshold=0.06)
        except Exception:
            self._appearance_encoder = PlayerAppearanceEncoder()
            self._reid = PlayerReIdentifier()

    def update(self, detections: np.ndarray, frame_number: int,
               frame: Optional[np.ndarray] = None) -> List[Dict]:
        """Update tracker with YOLO detections.

        Args:
            detections: N×6 array [x1, y1, x2, y2, conf, cls]
            frame_number: current frame number

        Returns:
            List of dicts with track_id, x, y, bbox for each tracked player
        """
        if len(detections) == 0:
            return []

        # Convert to supervision Detections format
        xyxy = detections[:, :4]
        confidence = detections[:, 4]
        class_id = detections[:, 5].astype(int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        # Run ByteTrack — gives stable tracker IDs
        tracked = self._byte_track.update_with_detections(sv_detections)

        positions = []
        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            track_id = int(tracked.tracker_id[i])

            # Use bottom-center of bbox as foot position
            cx = (x1 + x2) / 2.0
            cy_foot = y2

            try:
                # Use CameraModel.project_to_ground if available, else CourtCalibration.pixel_to_court
                if hasattr(self.calibration, 'project_to_ground'):
                    court_x, court_y = self.calibration.project_to_ground(cx, cy_foot)
                else:
                    court_x, court_y = self.calibration.pixel_to_court(cx, cy_foot)
            except Exception:
                continue

            # Filter detections outside court bounds
            if (court_x < self.COURT_X_MIN or court_x > self.COURT_X_MAX or
                    court_y < self.COURT_Y_MIN or court_y > self.COURT_Y_MAX):
                continue

            self._tracks[track_id] = {
                "x": float(court_x), "y": float(court_y),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "frame": frame_number,
            }
            positions.append({
                "track_id": track_id,
                "x": float(court_x), "y": float(court_y),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        if frame is not None:
            active_ids = {self._player_map[p["track_id"]] for p in positions
                          if p["track_id"] in self._player_map}
            for position in positions:
                track_id = position["track_id"]
                embedding = self._appearance_encoder.encode(frame, position["bbox"])
                if embedding is None:
                    continue
                player_id = self._player_map.get(track_id)
                if player_id is not None:
                    self._reid.register(player_id, embedding)
                    continue
                match = self._reid.match(embedding, excluded_players=active_ids)
                if match.confident and match.player_id is not None:
                    self._player_map[track_id] = match.player_id
                    active_ids.add(match.player_id)

        return positions

    def assign_player(self, track_id: int, player_id: str) -> None:
        self._player_map[track_id] = player_id

    def get_player_id(self, track_id: int) -> Optional[str]:
        return self._player_map.get(track_id)

    def get_player_position(self, player_id: str) -> Optional[Dict]:
        for tid, pid in self._player_map.items():
            if pid == player_id and tid in self._tracks:
                return self._tracks[tid]
        return None

    def find_closest_player(self, x: float, y: float) -> Optional[str]:
        min_dist = float("inf")
        closest = None
        for tid, pid in self._player_map.items():
            if tid not in self._tracks:
                continue
            track = self._tracks[tid]
            dist = np.sqrt((track["x"] - x) ** 2 + (track["y"] - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = pid
        return closest
