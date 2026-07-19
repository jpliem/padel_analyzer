"""Translate legacy CV events into semantic observations for padel rules."""

from typing import Dict, List, Optional

from models.observations import CourtSurface, ObservationKind, PadelObservation
from models.types import EventType, MatchEvent, TeamId


class SemanticEventBridge:
    """Stateful adapter; it describes evidence but never changes the score."""

    def __init__(self, team_map: Dict[str, int]):
        self.team_map = team_map
        self.point_started = False

    @staticmethod
    def _team(value) -> Optional[TeamId]:
        if isinstance(value, TeamId):
            return value
        if value in (1, "1"):
            return TeamId.TEAM_A
        if value in (2, "2"):
            return TeamId.TEAM_B
        return None

    @staticmethod
    def _side(event: MatchEvent) -> str:
        return str(event.metadata.get("side") or
                   ("near" if event.position.y < 10.0 else "far"))

    def translate(self, event: MatchEvent, server_team,
                  confidence: float) -> List[PadelObservation]:
        observations: List[PadelObservation] = []
        event_type = event.event_type
        team = self._team(server_team)

        if event_type in (EventType.SERVE, EventType.FAULT) and not self.point_started:
            observations.append(PadelObservation(
                kind=ObservationKind.POINT_READY,
                timestamp=event.timestamp,
                frame_number=event.frame_number,
                confidence=confidence,
                team=team,
                side="near" if team == TeamId.TEAM_A else "far",
                metadata={"source": "legacy_cv_bridge"},
            ))

        base = dict(timestamp=event.timestamp, frame_number=event.frame_number,
                    confidence=confidence,
                    position=(event.position.x, event.position.y, 0.0))
        if event_type == EventType.SERVE:
            observations.extend([
                PadelObservation(kind=ObservationKind.SERVE_STRUCK, team=team,
                                 metadata={"source": "legacy_serve_detector"}, **base),
                PadelObservation(
                    kind=ObservationKind.SURFACE_CONTACT,
                    surface=CourtSurface.FLOOR, side=self._side(event),
                    metadata={"correct_service_box": True,
                              "source": "legacy_serve_detector"}, **base),
            ])
        elif event_type == EventType.FAULT:
            observations.append(PadelObservation(
                kind=ObservationKind.SURFACE_CONTACT,
                surface=CourtSurface.FLOOR, side=self._side(event),
                metadata={"correct_service_box": False,
                          "detail": event.metadata.get("detail"),
                          "source": "legacy_serve_detector"}, **base))
        elif event_type == EventType.BOUNCE:
            observations.append(PadelObservation(
                kind=ObservationKind.SURFACE_CONTACT,
                surface=CourtSurface.FLOOR, side=self._side(event),
                metadata={"source": "legacy_bounce_detector"}, **base))
        elif event_type == EventType.HIT:
            track_id = event.metadata.get("track_id")
            player_id = event.metadata.get("player_id")
            player_team = self._team(self.team_map.get(player_id)) if player_id else None
            observations.append(PadelObservation(
                kind=ObservationKind.PLAYER_HIT, player_id=player_id,
                team=player_team, side=self._side(event),
                metadata={"track_id": track_id,
                          "source": "legacy_last_hitter_detector"}, **base))
        elif event_type == EventType.WALL_HIT:
            surface_name = str(event.metadata.get("surface_type", "glass")).lower()
            surface = CourtSurface.FENCE if "fence" in surface_name else CourtSurface.GLASS
            observations.append(PadelObservation(
                kind=ObservationKind.SURFACE_CONTACT, surface=surface,
                side=self._side(event), metadata={**event.metadata,
                                                  "source": "legacy_wall_detector"},
                **base))
        elif event_type == EventType.NET_HIT:
            observations.append(PadelObservation(
                kind=ObservationKind.SURFACE_CONTACT, surface=CourtSurface.NET,
                side=self._side(event), metadata={"source": "legacy_net_detector"},
                **base))
        return observations

    def point_ended(self) -> None:
        self.point_started = False

    def point_ready_accepted(self) -> None:
        self.point_started = True
