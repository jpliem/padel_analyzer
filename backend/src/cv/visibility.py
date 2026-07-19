"""Explicit visibility state for a ball that may disappear temporarily."""

from dataclasses import dataclass

from models.observations import BallVisibility


@dataclass(frozen=True)
class VisibilityEstimate:
    state: BallVisibility
    missing_frames: int
    confidence: float


class BallVisibilityTracker:
    def __init__(self) -> None:
        self.state = BallVisibility.UNKNOWN
        self.missing_frames = 0

    def update(self, *, detected: bool, detection_confidence: float = 0.0,
               occluded: bool = False, outside_fov: bool = False,
               outside_court_recoverable: bool = False) -> VisibilityEstimate:
        if detected:
            self.state = BallVisibility.VISIBLE
            self.missing_frames = 0
            confidence = detection_confidence
        else:
            self.missing_frames += 1
            if outside_court_recoverable:
                self.state = BallVisibility.OUTSIDE_COURT_RECOVERABLE
                confidence = 0.8
            elif occluded:
                self.state = BallVisibility.OCCLUDED
                confidence = 0.75
            elif outside_fov:
                self.state = BallVisibility.OUTSIDE_FOV
                confidence = 0.75
            else:
                self.state = BallVisibility.UNKNOWN
                confidence = max(0.1, 0.6 - 0.03 * self.missing_frames)
        return VisibilityEstimate(self.state, self.missing_frames, confidence)

    def reset(self) -> None:
        self.state = BallVisibility.UNKNOWN
        self.missing_frames = 0

