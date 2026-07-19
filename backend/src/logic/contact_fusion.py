"""Fuse independent evidence into contact proposals; never changes the score."""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ContactEvidence:
    frame_number: int
    player_id: Optional[str] = None
    audio_confidence: float = 0.0
    direction_change_confidence: float = 0.0
    racket_proximity_confidence: float = 0.0
    surface_proximity_confidence: float = 0.0


@dataclass(frozen=True)
class ContactProposal:
    contact_type: str
    player_id: Optional[str]
    confidence: float
    requires_review: bool
    evidence: Tuple[str, ...]


def fuse_contact(evidence: ContactEvidence, review_threshold: float = 0.68) -> ContactProposal:
    scores = {
        "audio": max(0.0, min(1.0, evidence.audio_confidence)),
        "direction_change": max(0.0, min(1.0, evidence.direction_change_confidence)),
        "racket_proximity": max(0.0, min(1.0, evidence.racket_proximity_confidence)),
        "surface_proximity": max(0.0, min(1.0, evidence.surface_proximity_confidence)),
    }
    if evidence.player_id and scores["racket_proximity"] >= scores["surface_proximity"]:
        kind = "racket_hit"
        confidence = (0.20 * scores["audio"] + 0.45 * scores["direction_change"] +
                      0.35 * scores["racket_proximity"])
    else:
        kind = "surface_contact"
        confidence = (0.20 * scores["audio"] + 0.45 * scores["direction_change"] +
                      0.35 * scores["surface_proximity"])
    used = tuple(name for name, value in scores.items() if value > 0)
    # At least two independent signals are required for automatic acceptance.
    review = confidence < review_threshold or len(used) < 2
    return ContactProposal(kind, evidence.player_id, confidence, review, used)

