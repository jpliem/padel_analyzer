from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class TeamId(Enum):
    TEAM_A = 1
    TEAM_B = 2


class PointReason(Enum):
    WINNER = "winner"
    DOUBLE_FAULT = "double_fault"
    OUT = "out"
    NET = "net"
    DOUBLE_BOUNCE = "double_bounce"
    WALL_BEFORE_BOUNCE = "wall_before_bounce"
    MANUAL = "manual"


class MatchFormat(Enum):
    BEST_OF_3 = 2
    BEST_OF_1 = 1


class EventType(Enum):
    BOUNCE = "BOUNCE"
    SERVE = "SERVE"
    FAULT = "FAULT"
    LET = "LET"
    NET_HIT = "NET_HIT"
    WALL_HIT = "WALL_HIT"
    OUT = "OUT"
    POINT_END = "POINT_END"
    HIT = "HIT"


class MatchState(Enum):
    IDLE = "IDLE"
    SERVING_1ST = "SERVING_1ST"
    SERVING_2ND = "SERVING_2ND"
    RALLY = "RALLY"
    POINT_ENDED = "POINT_ENDED"
    SCORE_UPDATE = "SCORE_UPDATE"


@dataclass
class ServerInfo:
    team_id: TeamId = TeamId.TEAM_A
    player_id: str = "P1"


@dataclass
class MatchConfig:
    match_name: str = "Match"
    players: Dict[str, str] = field(default_factory=lambda: {
        "P1": "Player 1", "P2": "Player 2",
        "P3": "Player 3", "P4": "Player 4",
    })
    teams: Dict[TeamId, List[str]] = field(default_factory=lambda: {
        TeamId.TEAM_A: ["P1", "P2"],
        TeamId.TEAM_B: ["P3", "P4"],
    })
    golden_point: bool = True
    format: MatchFormat = MatchFormat.BEST_OF_3
    first_server: ServerInfo = field(default_factory=ServerInfo)


@dataclass
class CourtPoint:
    x: float
    y: float


@dataclass
class BallPosition:
    x: float
    y: float
    z: float = 0.0
    speed: float = 0.0
    timestamp: float = 0.0


@dataclass
class PlayerPosition:
    player_id: str
    x: float
    y: float
    timestamp: float = 0.0


@dataclass
class MatchEvent:
    event_type: EventType
    timestamp: float
    frame_number: int
    position: CourtPoint
    confidence: float = 0.0
    metadata: Dict = field(default_factory=dict)
