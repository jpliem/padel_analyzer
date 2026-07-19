from __future__ import annotations

from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


ShortText = Annotated[str, Field(max_length=240)]
NoteText = Annotated[str, Field(max_length=360)]


class StrictModel(BaseModel):
    """Response base that keeps constrained decoders inside known fields."""

    model_config = ConfigDict(extra="forbid")


class TacticalPhase(StrictModel):
    phase: Literal[
        "team_a_attacking", "team_b_attacking", "balanced",
        "transition", "unclear",
    ] = "unclear"
    evidence: ShortText = ""


class CoachingObservation(StrictModel):
    team: Literal["team_a", "team_b", "both", "unclear"] = "unclear"
    category: Literal[
        "positioning", "partner_spacing", "net_control", "serve_return",
        "shot_selection", "movement", "communication", "other",
    ] = "other"
    observation: ShortText
    evidence_frame: Optional[int] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class RallyEnding(StrictModel):
    type: Literal[
        "winner", "forced_error", "unforced_error", "net", "out",
        "double_bounce", "unknown",
    ] = "unknown"
    likely_winner: Literal["team_a", "team_b", "unknown"] = "unknown"
    explanation: ShortText = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class RallyAnalysis(StrictModel):
    summary: NoteText
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rally_quality: Literal["low", "medium", "high", "unclear"] = "unclear"
    highlight_score: float = Field(default=0.0, ge=0.0, le=1.0)
    ending: RallyEnding = Field(default_factory=RallyEnding)
    tactical_phases: List[TacticalPhase] = Field(default_factory=list, max_length=4)
    coaching_observations: List[CoachingObservation] = Field(
        default_factory=list, max_length=4
    )
    uncertainty: List[ShortText] = Field(default_factory=list, max_length=4)


class TrainingPriority(StrictModel):
    title: ShortText
    reason: NoteText
    evidence_rallies: List[int] = Field(default_factory=list, max_length=6)
    suggested_drill: NoteText


class MatchStory(StrictModel):
    headline: ShortText
    overview: NoteText
    team_a_story: NoteText
    team_b_story: NoteText
    momentum_notes: List[ShortText] = Field(default_factory=list, max_length=6)
    training_priorities: List[TrainingPriority] = Field(
        default_factory=list, max_length=4
    )
    best_rallies: List[int] = Field(default_factory=list, max_length=6)
    limitations: List[ShortText] = Field(default_factory=list, max_length=6)


class BallPoint(StrictModel):
    """VLM pointing output: ball location in 0-1000 normalized coordinates."""

    found: bool = False
    x: int = Field(default=0, ge=0, le=1000)
    y: int = Field(default=0, ge=0, le=1000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class TrackAuditVerdict(StrictModel):
    """VLM judgement of one annotated tracker frame."""

    marker_on_ball: Literal["yes", "close", "no", "no_ball_visible", "unclear"] = "unclear"
    ball_visible_elsewhere: bool = False
    ball_location_hint: ShortText = ""
    note: ShortText = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def model_schema(model_type):
    if hasattr(model_type, "model_json_schema"):
        return model_type.model_json_schema()
    return model_type.schema()


def model_validate(model_type, value):
    if hasattr(model_type, "model_validate"):
        return model_type.model_validate(value)
    return model_type.parse_obj(value)


def model_dump(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()
