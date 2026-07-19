"""Select the active rally ball from detector candidates.

The detector answers "ball-like pixels".  This module answers the different
question "which candidate is consistent with the ball already in play?".
"""

from dataclasses import dataclass
from math import hypot
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class ActiveBallSelection:
    bbox: Optional[List[float]]
    confidence: float
    state: str
    candidate_count: int
    rejected_count: int
    reason: str

    def as_dict(self) -> Dict:
        return {
            "confidence": round(self.confidence, 4),
            "state": self.state,
            "candidate_count": self.candidate_count,
            "rejected_count": self.rejected_count,
            "reason": self.reason,
        }


class ActiveBallSelector:
    """Temporal candidate gating for spare balls and transient false positives."""

    def __init__(self, max_jump_px: float = 180.0, reacquire_after: int = 8):
        self.max_jump_px = max_jump_px
        self.reacquire_after = reacquire_after
        self._center: Optional[Tuple[float, float]] = None
        self._velocity = (0.0, 0.0)
        self._misses = 0
        self.total_candidates = 0
        self.total_rejected = 0
        self.uncertain_frames = 0
        self.last_selection = ActiveBallSelection(None, 0.0, "searching", 0, 0, "no candidates")

    @staticmethod
    def _center_of(candidate: Dict) -> Tuple[float, float]:
        x1, y1, x2, y2 = candidate["bbox"]
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def select(self, candidates: Sequence[Dict], frame_shape=None) -> ActiveBallSelection:
        usable = [c for c in candidates if len(c.get("bbox", [])) == 4]
        self.total_candidates += len(usable)
        if not usable:
            self._misses += 1
            selection = ActiveBallSelection(None, 0.0, "occluded", 0, 0, "detector miss")
            self.last_selection = selection
            return selection

        if self._center is None or self._misses >= self.reacquire_after:
            chosen = max(usable, key=lambda c: float(c.get("confidence", 0.0)))
            center = self._center_of(chosen)
            ambiguous = len(usable) > 1
            confidence = float(chosen.get("confidence", 0.0)) * (0.65 if ambiguous else 0.85)
            self._center = center
            self._velocity = (0.0, 0.0)
            self._misses = 0
            rejected = len(usable) - 1
            self.total_rejected += rejected
            if ambiguous:
                self.uncertain_frames += 1
            selection = ActiveBallSelection(
                list(chosen["bbox"]), confidence,
                "uncertain" if ambiguous else "acquired", len(usable), rejected,
                "multiple untracked ball-like objects" if ambiguous else "initial acquisition",
            )
            self.last_selection = selection
            return selection

        predicted = (self._center[0] + self._velocity[0], self._center[1] + self._velocity[1])
        ranked = []
        resolution_scale = 1.0
        if frame_shape is not None and len(frame_shape) >= 2:
            height, width = frame_shape[:2]
            resolution_scale = max(0.5, hypot(width, height) / hypot(1280, 720))
        allowed_jump = (self.max_jump_px * resolution_scale *
                        min(2.0, 1.0 + self._misses * 0.18))
        for candidate in usable:
            center = self._center_of(candidate)
            distance = hypot(center[0] - predicted[0], center[1] - predicted[1])
            detector_conf = float(candidate.get("confidence", 0.0))
            continuity = max(0.0, 1.0 - distance / allowed_jump)
            ranked.append((0.65 * continuity + 0.35 * detector_conf, distance, candidate, center))

        ranked.sort(key=lambda item: item[0], reverse=True)
        score, distance, chosen, center = ranked[0]
        if distance > allowed_jump:
            self._misses += 1
            self.total_rejected += len(usable)
            self.uncertain_frames += 1
            selection = ActiveBallSelection(
                None, 0.0, "uncertain", len(usable), len(usable),
                "all candidates violate temporal motion gate",
            )
            self.last_selection = selection
            return selection

        observed_velocity = (center[0] - self._center[0], center[1] - self._center[1])
        self._velocity = (
            0.55 * self._velocity[0] + 0.45 * observed_velocity[0],
            0.55 * self._velocity[1] + 0.45 * observed_velocity[1],
        )
        self._center = center
        self._misses = 0
        rejected = len(usable) - 1
        self.total_rejected += rejected
        confidence = min(1.0, max(0.0, score))
        state = "tracked" if confidence >= 0.55 else "uncertain"
        if state == "uncertain":
            self.uncertain_frames += 1
        selection = ActiveBallSelection(
            list(chosen["bbox"]), confidence, state, len(usable), rejected,
            "temporal continuity" if rejected else "single consistent candidate",
        )
        self.last_selection = selection
        return selection

    def diagnostics(self) -> Dict:
        return {
            "total_candidates": self.total_candidates,
            "rejected_candidates": self.total_rejected,
            "uncertain_frames": self.uncertain_frames,
            "last_selection": self.last_selection.as_dict(),
        }
