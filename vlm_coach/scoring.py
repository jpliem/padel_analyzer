from __future__ import annotations

from typing import List, Literal

from pydantic import Field, model_validator

from .schemas import ShortText, StrictModel
from .video import Segment


class RallyScout(StrictModel):
    """Cheap sequential classification; deliberately contains no score fields."""

    phase_at_start: Literal[
        "waiting", "serve_preparation", "serve_started", "rally", "point_ending", "unclear",
    ]
    phase_at_end: Literal[
        "waiting", "serve_preparation", "serve_started", "rally", "point_ending", "unclear",
    ]
    rally_visible: bool
    serve_candidate: bool
    point_end_candidate: bool
    evidence_frames: List[int] = Field(max_length=4)
    explanation: ShortText
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def reject_unsupported_activity(self):
        if self.confidence < 0.55:
            self.rally_visible = False
            self.serve_candidate = False
            self.point_end_candidate = False
        elif not self.evidence_frames:
            # A scout may cheaply forward a high-confidence active-play window
            # without citations. Specific serve/end claims still require them.
            self.serve_candidate = False
            self.point_end_candidate = False
        return self


class RallyVerdict(StrictModel):
    """One completed candidate rally; Python remains authoritative for scoring."""

    valid_rally: bool
    point_ended: bool
    winner: Literal["team_a", "team_b", "unknown"]
    decisive_team: Literal["team_a", "team_b", "unknown"]
    decisive_outcome: Literal["won_point", "lost_point", "unclear"]
    ending: Literal[
        "winner", "forced_error", "unforced_error", "net", "out",
        "double_bounce", "unknown",
    ]
    evidence_frames: List[int] = Field(max_length=5)
    explanation: ShortText
    confidence: float = Field(ge=0.0, le=1.0)
    review_required: bool

    @model_validator(mode="after")
    def reject_unsupported_winner(self):
        if not self.valid_rally:
            self.point_ended = False
        if not self.point_ended:
            self.ending = "unknown"
        derived_winner = "unknown"
        if self.decisive_team != "unknown" and self.decisive_outcome != "unclear":
            if self.decisive_outcome == "won_point":
                derived_winner = self.decisive_team
            else:
                derived_winner = (
                    "team_b" if self.decisive_team == "team_a" else "team_a"
                )
        if derived_winner == "unknown" or self.winner != derived_winner:
            self.winner = "unknown"
            self.review_required = True
        supported = (
            self.valid_rally and self.point_ended and self.winner != "unknown"
            and self.confidence >= 0.75 and bool(self.evidence_frames)
        )
        if not supported:
            self.winner = "unknown"
            self.review_required = True
        return self


class GapDecision(StrictModel):
    """Does a low-motion gap separate points or occur inside one rally?"""

    middle_state: Literal["active_rally", "between_points", "unclear"]
    evidence_frames: List[int] = Field(max_length=4)
    explanation: ShortText
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def reject_unsupported_state(self):
        if self.confidence < 0.7 or not self.evidence_frames:
            self.middle_state = "unclear"
        return self

    @property
    def same_rally_continues(self) -> bool:
        # Unclear gaps merge conservatively; only reviewed evidence may split.
        return self.middle_state != "between_points"


class RollingObservation(StrictModel):
    """One chronological window in a stateful, continuous match scan."""

    phase_at_start: Literal[
        "between_points", "serve_preparation", "serve_started", "rally", "point_ending", "unclear",
    ]
    phase_at_end: Literal[
        "between_points", "serve_preparation", "serve_started", "rally", "point_ending", "unclear",
    ]
    relation_to_previous: Literal[
        "same_rally", "new_rally", "no_active_rally", "unclear",
    ]
    rally_visible: bool
    point_end_candidate: bool
    evidence_frames: List[int] = Field(max_length=4)
    active_play_frames: List[int] = Field(max_length=4)
    reset_frames: List[int] = Field(max_length=4)
    observed_change: ShortText
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def reject_unsupported_transition(self):
        if self.confidence < 0.6:
            self.relation_to_previous = "unclear"
            self.point_end_candidate = False
            self.rally_visible = False
        if self.rally_visible and not self.active_play_frames:
            self.rally_visible = False
        ordered_transition = (
            bool(self.active_play_frames) and bool(self.reset_frames)
            and max(self.active_play_frames) < min(self.reset_frames)
        )
        if self.point_end_candidate and not ordered_transition:
            self.point_end_candidate = False
        return self


class FrameState(StrictModel):
    index: int = Field(ge=0)
    state: Literal["active_play", "reset", "unclear"]
    confidence: float = Field(ge=0.0, le=1.0)


class FrameStateObservation(StrictModel):
    frames: List[FrameState] = Field(max_length=16)


FRAME_STATE_PROMPT = """Classify each supplied padel image independently. Images are
chronological and visibly labelled FRAME 00, FRAME 01, and so on. Return one entry
for every supplied frame in increasing index order.

Use only:
- active_play: visible athletic rally action or clearly engaged ready movement.
- reset: relaxed walking, dead-ball retrieval, celebration, waiting, or serve setup.
- unclear: one still image cannot distinguish active play from waiting.

The tiny ball need not be visible. Do not infer an ending, winner, score, or story.
Do not copy a label from adjacent frames merely for consistency. Return schema JSON.
"""


def derive_frame_state_window(payload: dict, frame_count: int) -> dict:
    """Turn independent frame labels into conservative temporal evidence."""
    by_index = {}
    for item in payload.get("frames", []):
        index = item.get("index")
        if isinstance(index, int) and 0 <= index < frame_count and index not in by_index:
            by_index[index] = item
    active = sorted(
        index for index, item in by_index.items()
        if item.get("state") == "active_play" and float(item.get("confidence", 0)) >= 0.7
    )
    reset = sorted(
        index for index, item in by_index.items()
        if item.get("state") == "reset" and float(item.get("confidence", 0)) >= 0.7
    )
    ending = False
    ending_active = []
    ending_reset = []
    for split in range(1, frame_count):
        before = [index for index in active if index < split]
        after = [index for index in reset if index >= split]
        if len(before) >= 2 and len(after) >= 2:
            ending = True
            ending_active = before[-2:]
            ending_reset = after[:2]
            break
    return {
        "frames_received": len(by_index),
        "coverage_complete": len(by_index) == frame_count,
        "rally_visible": len(active) >= 2,
        "point_end_candidate": ending,
        "active_play_frames": ending_active if ending else active[:4],
        "reset_frames": ending_reset if ending else reset[:4],
    }


def sanitize_frame_evidence(decision: dict, frame_count: int,
                            strict: bool = True) -> dict:
    """Reject VLM citations to frames that were never supplied."""
    invalid = False
    for key in ("evidence_frames", "active_play_frames", "reset_frames"):
        if key not in decision:
            continue
        original = decision.get(key, [])
        valid = [item for item in original
                 if isinstance(item, int) and 0 <= item < frame_count]
        decision[key] = valid
        invalid = invalid or len(valid) != len(original)
    if invalid and strict:
        decision["confidence"] = min(float(decision.get("confidence", 0.0)), 0.49)
        if "serve_candidate" in decision:
            decision["rally_visible"] = False
            decision["serve_candidate"] = False
            decision["point_end_candidate"] = False
        if "winner" in decision:
            decision["winner"] = "unknown"
            decision["review_required"] = True
        if "middle_state" in decision:
            decision["middle_state"] = "unclear"
    return decision


SCOUT_PROMPT = """You are a conservative padel rally boundary detector, not a commentator.
The fixed camera is behind one baseline. Lower/larger players are on the NEAR
side; upper/smaller players are on the FAR side. Team mapping for this window:
NEAR={near_team}; FAR={far_team}. Images are chronological, zero-based frames.

Previous window phase: {previous_phase}.

Classify only visible play state. A window may begin in the middle of a rally; a
visible serve is not required. Athletic ready positions plus changing player
positions, racket swings, reaches, volleys, or dives across frames support a rally
even when the tiny ball is invisible. Relaxed walking, waiting, or a reset is not
a rally. A missing ball is not evidence that a point ended. Mark point_end_candidate only when active play
changes to a clear stop/reset. Never calculate or guess the score or point winner.
Any true flag requires confidence of at least 0.55. Serve and point-ending flags
also require evidence_frames. A high-confidence rally_visible flag may omit frame
citations because it only forwards the clip to a stricter judge. Use one short
evidence sentence and return only schema JSON.
"""


VERDICT_PROMPT = """You are adjudicating exactly one candidate padel rally from a fixed camera
behind one baseline. Lower/larger players are NEAR; upper/smaller players are FAR.
NEAR={near_team}; FAR={far_team}. Images are chronological, zero-based frames and
include context before the suspected start and after the suspected ending.

Known score before this candidate: {score_before}.
Boundary scout evidence: {scout_evidence}

First decide whether sustained play actually occurred and visibly ended. Name a
winner only when the images support which team made the final successful play or
failed return. Player position, a missing tiny ball, or one gesture alone is not
enough. Winner MUST be unknown when valid_rally is false, point_ended is false,
confidence is below 0.75, or evidence_frames is empty. When evidence is incomplete,
review_required must be true. Never calculate the next score. Cite frames and return
only schema JSON.

Cross-check the decisive action explicitly:
- If Team A hits out, decisive_team=team_a, decisive_outcome=lost_point, winner=team_b.
- If Team A hits a winner, decisive_team=team_a, decisive_outcome=won_point, winner=team_a.
- If that relationship is unclear, use unknown/unclear and winner=unknown.
"""


GAP_PROMPT = """You are examining a low-motion gap between two active padel video bursts.
The fixed camera is behind one baseline. Lower/larger players are NEAR={near_team};
upper/smaller players are FAR={far_team}. Images cover two seconds before the gap,
the gap, and two seconds after it, in chronological zero-based order. Refer to
images only as frame 0, frame 1, and so on; never use video timestamps.

Answer one narrow question: is this a pause inside the SAME rally, or a boundary
between points? Active play at BOTH the beginning and end is expected and does not
prove it is one rally. Inspect the MIDDLE frames first. A point boundary exists if
the middle shows play stop/reset and the ending frames show preparation for a new
serve or a new rally. Low movement, defensive waiting, a lob, or players holding
ready positions throughout means the same rally continues. A reset includes relaxed
walking, retrieving a dead ball, returning to service/receive positions, or preparing
a new serve after play stopped. Do not identify a winner.

Classify only the MIDDLE frames:
- active_rally: players remain in athletic play/ready positions within one point.
- between_points: play stopped; players retrieve/reset/walk or prepare the next serve.
- unclear: the middle does not prove either state.

The label requires confidence >= 0.70 and cited zero-based evidence frames;
otherwise use unclear. Return only schema JSON.
"""


ROLLING_PROMPT = """You are tracking a padel match through consecutive overlapping image
windows. The fixed camera is behind one baseline. Lower/larger players are
NEAR={near_team}; upper/smaller players are FAR={far_team}.

The images in THIS request are strictly chronological. Their visible FRAME labels
and the timeline below define the order. Several opening images may overlap the
previous window. Do not treat an overlap image as a new event.

Previous machine state (context only, never visual evidence):
{previous_state}

Decide how visible play changes through THIS window. relation_to_previous means:
- same_rally: visible action continues the rally active at the previous window end.
- new_rally: a stop/reset or serve preparation proves a different rally began.
- no_active_rally: players are waiting/resetting and no rally is visible.
- unclear: images do not prove the relationship.

A missing tiny ball never proves an ending. Cite active_play_frames for frames that
visibly show athletic play, and reset_frames for later frames that visibly show a
stop, relaxed walking, retrieval, celebration, or new-serve preparation. Mark
point_end_candidate only when both lists are non-empty and every cited reset frame
comes after the final cited active-play frame. evidence_frames may summarize the
strongest citations. Do not name a winner and do not calculate score. Previous state
helps continuity but cannot override current images. Return only schema JSON.
"""


def rolling_windows(duration: float, length: float = 12.0,
                    overlap: float = 3.0) -> list[Segment]:
    """Chronological windows with explicit overlap for continuity."""
    if duration <= 0 or length <= 0 or overlap < 0 or overlap >= length:
        return []
    return fixed_windows(duration, length=length, stride=length - overlap)


def rolling_context(previous: dict | None) -> str:
    """Bounded context: carry decisions, never an ever-growing transcript."""
    if not previous:
        return '{"phase_at_end":"between_points","active_candidate":false}'
    fields = (
        "phase_at_end", "relation_to_previous", "rally_visible",
        "point_end_candidate", "confidence",
    )
    import json
    return json.dumps({key: previous.get(key) for key in fields if key in previous},
                      separators=(",", ":"))


def reconcile_rolling_observation(previous: dict | None, current: dict) -> dict:
    """Downgrade transitions that contradict the preceding machine state."""
    relation = current.get("relation_to_previous")
    if current.get("rally_visible") and relation == "no_active_rally":
        current["relation_to_previous"] = "unclear"
        current["confidence"] = min(float(current.get("confidence", 0.0)), 0.59)
    if relation == "same_rally" and previous is None:
        current["relation_to_previous"] = "unclear"
        current["confidence"] = min(float(current.get("confidence", 0.0)), 0.59)
    elif previous and relation == "same_rally":
        previous_active = (
            previous.get("rally_visible")
            or previous.get("phase_at_end") in {"serve_started", "rally", "point_ending"}
        )
        if not previous_active:
            current["relation_to_previous"] = "unclear"
            current["confidence"] = min(float(current.get("confidence", 0.0)), 0.59)
    return current


def fixed_windows(duration: float, length: float = 6.0, stride: float = 3.0) -> list[Segment]:
    if duration <= 0 or length <= 0 or stride <= 0:
        return []
    windows = []
    start = 0.0
    while start < duration:
        end = min(duration, start + length)
        if end - start >= 1.0:
            windows.append(Segment(start, end))
        if end >= duration:
            break
        start += stride
    return windows


def merge_scout_windows(items: list[tuple[Segment, RallyScout]],
                        padding: float = 2.0) -> list[Segment]:
    """Merge sequential active/ending classifications into rally candidates."""
    active = [
        segment for segment, scout in items
        if scout.rally_visible or scout.serve_candidate or scout.point_end_candidate
    ]
    if not active:
        return []
    merged = [active[0]]
    for segment in active[1:]:
        previous = merged[-1]
        if segment.start <= previous.end + 0.25:
            merged[-1] = Segment(previous.start, max(previous.end, segment.end))
        else:
            merged.append(segment)
    return [Segment(max(0.0, item.start - padding), item.end + padding) for item in merged]


def add_scoring_context(segments: list[Segment], duration: float,
                        postroll: float = 3.0) -> list[Segment]:
    """Extend motion candidates far enough to show the ending/reset."""
    contextual = []
    for index, segment in enumerate(segments):
        end = min(duration, segment.end + postroll)
        if index + 1 < len(segments):
            end = min(end, max(segment.end, segments[index + 1].start - 0.1))
        contextual.append(Segment(segment.start, end))
    return contextual


def gap_contexts(segments: list[Segment], duration: float,
                 shoulder: float = 2.0) -> list[tuple[Segment, Segment]]:
    """Return (raw gap, visual context) for adjacent motion bursts."""
    result = []
    for previous, following in zip(segments, segments[1:]):
        if following.start <= previous.end:
            continue
        gap = Segment(previous.end, following.start)
        context = Segment(max(0.0, gap.start - shoulder),
                          min(duration, gap.end + shoulder))
        result.append((gap, context))
    return result


def merge_motion_bursts(segments: list[Segment],
                        decisions: list[GapDecision]) -> list[Segment]:
    """Merge bursts unless a reviewed gap decision proves a point boundary."""
    if not segments:
        return []
    merged = [segments[0]]
    for index, segment in enumerate(segments[1:]):
        decision = decisions[index] if index < len(decisions) else None
        if decision is None or decision.same_rally_continues:
            previous = merged[-1]
            merged[-1] = Segment(previous.start, segment.end)
        else:
            merged.append(segment)
    return merged
