from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class EventDetectorConfig:
    bounce_z_threshold: float = 0.3
    bounce_speed_dip_pct: float = 0.4
    serve_timeout_frames: int = 90
    winner_timeout_frames: int = 60
    ball_stopped_frames: int = 15
    auto_assign_after_frames: int = 30
    enclosure_bounds: Dict[str, float] = field(default_factory=lambda: {
        "x_min": -0.5, "x_max": 10.5,
        "y_min": -1.0, "y_max": 21.0,
    })
