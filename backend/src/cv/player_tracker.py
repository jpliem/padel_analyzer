import numpy as np
from typing import List, Dict, Optional
from cv.court_calibration import CourtCalibration


class PlayerTracker:
    def __init__(self, calibration: CourtCalibration):
        self.calibration = calibration
        self._tracks: Dict[int, Dict] = {}
        self._player_map: Dict[int, str] = {}
        self._next_track_id = 1
        self._prev_bboxes: Dict[int, np.ndarray] = {}

    # Enclosure bounds — players outside this area are filtered out
    COURT_X_MIN, COURT_X_MAX = -1.0, 11.0
    COURT_Y_MIN, COURT_Y_MAX = -2.0, 22.0
    MAX_ACTIVE_TRACKS = 4  # padel has exactly 4 players
    STALE_FRAMES = 30  # remove tracks not seen for this many frames

    def update(self, detections: np.ndarray, frame_number: int) -> List[Dict]:
        if len(detections) == 0:
            return []
        positions = []
        new_bboxes = {}
        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            cx = (x1 + x2) / 2.0
            cy_foot = y2  # bottom of bbox = feet
            court_x, court_y = self.calibration.pixel_to_court(cx, cy_foot)

            # Filter: skip detections outside court bounds (refs, spectators)
            if (court_x < self.COURT_X_MIN or court_x > self.COURT_X_MAX or
                    court_y < self.COURT_Y_MIN or court_y > self.COURT_Y_MAX):
                continue

            bbox = np.array([x1, y1, x2, y2])
            track_id = self._match_track(bbox)
            if track_id is None:
                # Only create new track if we haven't hit max
                if len(new_bboxes) >= self.MAX_ACTIVE_TRACKS:
                    continue
                track_id = self._next_track_id
                self._next_track_id += 1
            self._tracks[track_id] = {
                "x": float(court_x), "y": float(court_y),
                "bbox": bbox, "frame": frame_number,
            }
            new_bboxes[track_id] = bbox
            positions.append({
                "track_id": track_id,
                "x": float(court_x), "y": float(court_y),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        # Remove stale tracks not seen recently
        stale_ids = [tid for tid, t in self._tracks.items()
                     if frame_number - t["frame"] > self.STALE_FRAMES]
        for tid in stale_ids:
            del self._tracks[tid]
            self._prev_bboxes.pop(tid, None)

        self._prev_bboxes = new_bboxes
        return positions

    def _match_track(self, bbox: np.ndarray, iou_threshold: float = 0.3) -> Optional[int]:
        best_id = None
        best_iou = iou_threshold
        for tid, prev_bbox in self._prev_bboxes.items():
            iou = self._compute_iou(bbox, prev_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        return best_id

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

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
