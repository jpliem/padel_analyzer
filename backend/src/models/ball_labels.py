"""Versioned 2D ball-label contract used by training and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


SCHEMA_VERSION = "1.0"


class BallLabelState(str, Enum):
    UNREVIEWED = "unreviewed"
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    BLURRED = "blurred"
    OUTSIDE_FRAME = "outside_frame"
    HARD_NEGATIVE = "hard_negative"
    UNCERTAIN = "uncertain"


EVENT_TAGS = frozenset({
    "serve", "racket_hit", "bounce", "net", "glass", "fence",
    "gate_exit", "outside_return", "point_end",
})


@dataclass(frozen=True)
class BallFrameLabel:
    frame: int
    timestamp: float
    image: str
    state: BallLabelState
    center: Optional[Tuple[float, float]] = None
    event_tags: Tuple[str, ...] = ()
    sequence_id: str = ""
    camera_id: str = ""
    split: Optional[str] = None
    metadata: Dict = field(default_factory=dict)


def validate_label_document(doc: Dict) -> List[str]:
    """Return all validation errors; an empty list means training-safe."""
    errors: List[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION!r}")
    if doc.get("coordinate_space") != "original_video_pixels":
        errors.append("coordinate_space must be 'original_video_pixels'")
    labels = doc.get("labels")
    if not isinstance(labels, list) or not labels:
        return errors + ["labels must be a non-empty list"]

    seen = set()
    for i, item in enumerate(labels):
        prefix = f"labels[{i}]"
        if not isinstance(item, dict):
            errors.append(f"{prefix} must be an object")
            continue
        frame = item.get("frame")
        if not isinstance(frame, int) or frame < 0:
            errors.append(f"{prefix}.frame must be a non-negative integer")
        key = (item.get("camera_id", doc.get("camera_id")), frame)
        if key in seen:
            errors.append(f"{prefix} duplicates camera/frame {key!r}")
        seen.add(key)
        try:
            state = BallLabelState(item.get("state", ""))
        except ValueError:
            errors.append(f"{prefix}.state is invalid")
            continue
        center = item.get("center")
        if state in (BallLabelState.VISIBLE, BallLabelState.BLURRED):
            if (not isinstance(center, list) or len(center) != 2 or
                    not all(isinstance(v, (int, float)) for v in center)):
                errors.append(f"{prefix}.center is required for {state.value}")
        elif center is not None:
            errors.append(f"{prefix}.center must be null for {state.value}")
        for tag in item.get("event_tags", []):
            if tag not in EVENT_TAGS:
                errors.append(f"{prefix}.event_tags contains unknown tag {tag!r}")
        split = item.get("split")
        if split is not None and split not in ("train", "val", "test"):
            errors.append(f"{prefix}.split must be train, val, or test")
    return errors


def group_safe_split(sequence_ids: Iterable[str]) -> Dict[str, str]:
    """Deterministically split whole rallies/sequences, preventing frame leak."""
    import hashlib

    result = {}
    for sequence_id in sorted(set(sequence_ids)):
        bucket = int(hashlib.sha256(sequence_id.encode("utf-8")).hexdigest()[:8], 16) % 100
        result[sequence_id] = "train" if bucket < 70 else "val" if bucket < 85 else "test"
    return result
