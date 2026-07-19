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
    out_of_court_play_enabled: bool = False


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


@dataclass
class CameraObservation:
    """Per-frame observation from a single camera."""
    camera_id: str
    ball_pixel: Optional[tuple] = None       # (px, py) raw pixel center
    ball_bbox: Optional[list] = None         # [x1, y1, x2, y2] raw pixel bbox
    ball_court: Optional[BallPosition] = None  # Projected to court coords
    confidence: float = 0.0
    player_detections: List[Dict] = field(default_factory=list)  # pixel + court
    timestamp: float = 0.0
    frame_number: int = 0


@dataclass
class WorldState:
    """Fused world state from all cameras for a single frame."""
    ball: Optional[BallPosition] = None
    ball_velocity: Optional[tuple] = None  # (vx, vy, vz) in m/s
    players: List[PlayerPosition] = field(default_factory=list)
    contributing_cameras: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    frame_number: int = 0


@dataclass
class WallHitMetadata:
    """Metadata for a wall hit event."""
    wall_id: str = ""
    surface_type: str = ""
    impact_point: tuple = (0.0, 0.0, 0.0)
    speed_at_impact: float = 0.0
    incoming_angle: float = 0.0
