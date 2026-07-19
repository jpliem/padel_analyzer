"""Semantic observations and decisions shared by perception and padel rules.

The computer-vision layer reports what it believes happened.  It must not
directly mutate the score.  :class:`logic.padel_rules.PadelRulesEngine` consumes
these observations and applies the rules of padel in their temporal order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Tuple

from models.types import TeamId


class BallVisibility(Enum):
    VISIBLE = "visible"
    OCCLUDED = "occluded"
    OUTSIDE_FOV = "outside_fov"
    OUTSIDE_COURT_RECOVERABLE = "outside_court_recoverable"
    UNKNOWN = "unknown"


class CourtSurface(Enum):
    FLOOR = "floor"
    GLASS = "glass"
    FENCE = "fence"
    NET = "net"
    NET_POST = "net_post"
    PLAYER = "player"
    RACKET = "racket"
    CEILING = "ceiling"
    EXTERNAL_OBJECT = "external_object"


class ObservationKind(Enum):
    POINT_READY = "point_ready"
    SERVE_STRUCK = "serve_struck"
    PLAYER_HIT = "player_hit"
    SURFACE_CONTACT = "surface_contact"
    BALL_EXITED = "ball_exited"
    VISIBILITY_CHANGED = "visibility_changed"
    INTERFERENCE = "interference"
    MANUAL_POINT = "manual_point"


class PointPhase(Enum):
    IDLE = "idle"
    FIRST_SERVE = "first_serve"
    SECOND_SERVE = "second_serve"
    RETURN_OF_SERVE = "return_of_serve"
    RALLY = "rally"
    OUTSIDE_PLAY = "outside_play"
    POINT_ENDED = "point_ended"


class DecisionKind(Enum):
    NONE = "none"
    SERVE_STARTED = "serve_started"
    SERVICE_FAULT = "service_fault"
    SERVICE_LET = "service_let"
    VALID_SERVE = "valid_serve"
    RALLY_CONTINUES = "rally_continues"
    POINT_LET = "point_let"
    POINT_AWARDED = "point_awarded"
    REVIEW_REQUIRED = "review_required"


@dataclass(frozen=True)
class PadelObservation:
    kind: ObservationKind
    timestamp: float
    frame_number: int = 0
    confidence: float = 1.0
    team: Optional[TeamId] = None
    player_id: Optional[str] = None
    side: Optional[str] = None  # "near" or "far"
    surface: Optional[CourtSurface] = None
    visibility: Optional[BallVisibility] = None
    position: Optional[Tuple[float, float, float]] = None
    metadata: Dict = field(default_factory=dict)


@dataclass(frozen=True)
class RuleDecision:
    kind: DecisionKind
    phase: PointPhase
    reason: str = ""
    winner: Optional[TeamId] = None
    confidence: float = 1.0
    requires_confirmation: bool = False
    evidence: Tuple[str, ...] = ()

