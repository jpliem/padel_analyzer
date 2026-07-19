"""Deterministic point-level rules for padel.

This module deliberately knows nothing about pixels, neural networks, or
detector thresholds.  It interprets ordered semantic observations and returns
decisions.  Low-confidence perception remains reviewable and a temporary loss
of visibility never ends a point by itself.
"""

from __future__ import annotations

from typing import List, Optional

from models.observations import (
    BallVisibility,
    CourtSurface,
    DecisionKind,
    ObservationKind,
    PadelObservation,
    PointPhase,
    RuleDecision,
)
from models.types import TeamId


def _opponent(team: TeamId) -> TeamId:
    return TeamId.TEAM_B if team == TeamId.TEAM_A else TeamId.TEAM_A


class PadelRulesEngine:
    """Apply FIP point rules to a stream of semantic observations.

    The engine covers the observable rules needed by the analyzer.  Referee-only
    judgements (receiver readiness, deliberate interference, safety-cord and
    equipment violations) enter as explicit observations or manual decisions.
    """

    def __init__(self, out_of_court_play_enabled: bool = False,
                 review_threshold: float = 0.65) -> None:
        self.out_of_court_play_enabled = out_of_court_play_enabled
        self.review_threshold = review_threshold
        self.phase = PointPhase.IDLE
        self.server_team: Optional[TeamId] = None
        self.serving_side: Optional[str] = None
        self.serve_attempt = 0
        self.last_hitter: Optional[TeamId] = None
        self.last_hit_side: Optional[str] = None
        self._serve_touched_net = False
        self._serve_landed = False
        self._serve_box_valid = False
        self._bounce_side: Optional[str] = None
        self._bounce_count = 0
        self._ball_crossed_after_hit = False
        self._evidence: List[str] = []

    def reset(self) -> None:
        self.__init__(
            out_of_court_play_enabled=self.out_of_court_play_enabled,
            review_threshold=self.review_threshold,
        )

    def process(self, obs: PadelObservation) -> RuleDecision:
        if obs.confidence < self.review_threshold:
            return self._decision(
                DecisionKind.REVIEW_REQUIRED,
                "low_confidence_observation",
                confidence=obs.confidence,
                requires_confirmation=True,
            )

        if obs.kind == ObservationKind.POINT_READY:
            return self._point_ready(obs)
        if obs.kind == ObservationKind.MANUAL_POINT:
            return self._manual_point(obs)
        if obs.kind == ObservationKind.INTERFERENCE:
            return self._interference(obs)
        if obs.kind == ObservationKind.VISIBILITY_CHANGED:
            return self._visibility(obs)

        if self.phase in (PointPhase.FIRST_SERVE, PointPhase.SECOND_SERVE,
                          PointPhase.RETURN_OF_SERVE):
            return self._process_service(obs)
        if self.phase in (PointPhase.RALLY, PointPhase.OUTSIDE_PLAY):
            return self._process_rally(obs)

        return self._decision(DecisionKind.NONE, "observation_ignored")

    def _point_ready(self, obs: PadelObservation) -> RuleDecision:
        if obs.team is None:
            return self._decision(
                DecisionKind.REVIEW_REQUIRED,
                "server_team_missing",
                requires_confirmation=True,
            )
        self.server_team = obs.team
        self.serving_side = obs.side
        self.serve_attempt = 1
        self.phase = PointPhase.FIRST_SERVE
        self.last_hitter = None
        self._reset_serve_sequence()
        self._reset_rally_sequence()
        self._evidence = [self._tag(obs)]
        return self._decision(DecisionKind.SERVE_STARTED, "first_service")

    def _process_service(self, obs: PadelObservation) -> RuleDecision:
        if obs.kind == ObservationKind.SERVE_STRUCK:
            if obs.team is not None and obs.team != self.server_team:
                return self._decision(
                    DecisionKind.REVIEW_REQUIRED,
                    "wrong_server_observed",
                    requires_confirmation=True,
                )
            self._evidence.append(self._tag(obs))
            return self._decision(DecisionKind.NONE, "serve_in_flight")

        if obs.kind == ObservationKind.PLAYER_HIT:
            return self._service_player_hit(obs)

        if obs.kind == ObservationKind.BALL_EXITED:
            if not self._serve_landed:
                return self._service_fault("serve_exited_before_legal_bounce")
            if obs.metadata.get("through_gate") and not self.out_of_court_play_enabled:
                return self._service_fault("serve_exited_gate_without_authorized_play")
            return self._complete_pending_serve("serve_exit_after_legal_bounce")

        if obs.kind != ObservationKind.SURFACE_CONTACT or obs.surface is None:
            return self._decision(DecisionKind.NONE, "service_observation_ignored")

        self._evidence.append(self._tag(obs))
        if obs.surface in (CourtSurface.NET, CourtSurface.NET_POST):
            if not self._serve_landed:
                self._serve_touched_net = True
            return self._decision(DecisionKind.NONE, "service_net_contact")

        if obs.surface == CourtSurface.FLOOR:
            if not self._serve_landed:
                correct_box = bool(obs.metadata.get("correct_service_box", False))
                if not correct_box:
                    return self._service_fault("serve_bounced_outside_correct_box")
                self._serve_landed = True
                self._serve_box_valid = True
                self._bounce_side = obs.side
                self._bounce_count = 1
                self.phase = PointPhase.RETURN_OF_SERVE
                return self._decision(DecisionKind.NONE, "serve_landed_pending_return")

            if obs.side == self._bounce_side:
                self._bounce_count += 1
                if self._serve_touched_net:
                    return self._service_let("net_service_before_second_bounce")
                return self._award(self.server_team, "receiver_failed_before_second_bounce")
            return self._decision(
                DecisionKind.REVIEW_REQUIRED,
                "unexpected_service_floor_contact",
                requires_confirmation=True,
            )

        if obs.surface == CourtSurface.FENCE:
            if self._serve_landed:
                return self._service_fault("serve_hit_fence_before_return")
            return self._service_fault("serve_hit_fence_before_bounce")

        if obs.surface == CourtSurface.GLASS:
            if not self._serve_landed:
                return self._service_fault("serve_hit_wall_before_bounce")
            return self._complete_pending_serve("legal_glass_after_service_bounce")

        if obs.surface in (CourtSurface.CEILING, CourtSurface.EXTERNAL_OBJECT):
            if self._serve_landed and self._serve_touched_net:
                return self._service_let("net_service_interrupted")
            return self._service_fault("illegal_service_contact")

        if obs.surface == CourtSurface.PLAYER:
            if self._serve_touched_net:
                return self._service_let("net_service_touched_player")
            if obs.team == _opponent(self.server_team):
                return self._award(self.server_team, "serve_hit_receiver_before_bounce")
            return self._service_fault("serve_hit_server_or_partner")

        return self._decision(DecisionKind.NONE, "service_continues")

    def _service_player_hit(self, obs: PadelObservation) -> RuleDecision:
        if obs.team is None:
            return self._decision(
                DecisionKind.REVIEW_REQUIRED,
                "hitter_team_missing",
                requires_confirmation=True,
            )
        if obs.team == self.server_team:
            return self._service_fault("server_team_touched_serve_again")
        if not self._serve_landed:
            if self._serve_touched_net:
                return self._service_let("net_service_touched_receiver")
            return self._award(self.server_team, "receiver_volleyed_serve")
        if self._serve_touched_net:
            return self._service_let("net_service")

        self.phase = PointPhase.RALLY
        self.last_hitter = obs.team
        self.last_hit_side = obs.side
        self._reset_rally_sequence()
        self._evidence.append(self._tag(obs))
        return self._decision(DecisionKind.VALID_SERVE, "legal_return_of_serve")

    def _complete_pending_serve(self, reason: str) -> RuleDecision:
        if self._serve_touched_net:
            return self._service_let("net_service")
        return self._decision(DecisionKind.NONE, reason)

    def _service_fault(self, reason: str) -> RuleDecision:
        if self.serve_attempt == 1:
            self.serve_attempt = 2
            self.phase = PointPhase.SECOND_SERVE
            self._reset_serve_sequence()
            return self._decision(DecisionKind.SERVICE_FAULT, reason)
        return self._award(_opponent(self.server_team), "double_fault:" + reason)

    def _service_let(self, reason: str) -> RuleDecision:
        self.phase = (PointPhase.FIRST_SERVE if self.serve_attempt == 1
                      else PointPhase.SECOND_SERVE)
        self._reset_serve_sequence()
        return self._decision(DecisionKind.SERVICE_LET, reason)

    def _process_rally(self, obs: PadelObservation) -> RuleDecision:
        if obs.kind == ObservationKind.PLAYER_HIT:
            return self._rally_hit(obs)
        if obs.kind == ObservationKind.BALL_EXITED:
            return self._rally_exit(obs)
        if obs.kind != ObservationKind.SURFACE_CONTACT or obs.surface is None:
            return self._decision(DecisionKind.NONE, "rally_observation_ignored")

        self._evidence.append(self._tag(obs))
        if obs.surface in (CourtSurface.NET, CourtSurface.NET_POST):
            return self._decision(DecisionKind.RALLY_CONTINUES, "legal_rally_net_contact")

        if obs.surface == CourtSurface.FLOOR:
            return self._rally_floor(obs)

        if obs.surface == CourtSurface.GLASS:
            if self.last_hitter is None:
                return self._review("glass_contact_without_known_hitter")
            if self._bounce_count == 0 and obs.side != self.last_hit_side:
                return self._award(_opponent(self.last_hitter), "opponent_wall_before_floor")
            return self._decision(DecisionKind.RALLY_CONTINUES, "legal_glass_contact")

        if obs.surface == CourtSurface.FENCE:
            if self.last_hitter is None:
                return self._review("fence_contact_without_known_hitter")
            if self._bounce_count == 0 and obs.side == self.last_hit_side:
                return self._award(_opponent(self.last_hitter), "own_fence_before_crossing")
            if self._bounce_count == 0:
                return self._award(_opponent(self.last_hitter), "opponent_fence_before_floor")
            return self._decision(DecisionKind.RALLY_CONTINUES, "legal_fence_after_bounce")

        if obs.surface in (CourtSurface.CEILING, CourtSurface.EXTERNAL_OBJECT):
            if self.last_hitter is None:
                return self._review("external_contact_without_known_hitter")
            if self._bounce_count >= 1:
                return self._award(self.last_hitter, "legal_bounce_then_unrecoverable_contact")
            return self._award(_opponent(self.last_hitter), "external_contact_before_legal_bounce")

        if obs.surface == CourtSurface.PLAYER:
            if obs.team is None:
                return self._review("player_contact_team_unknown")
            return self._award(_opponent(obs.team), "ball_touched_player")

        return self._decision(DecisionKind.RALLY_CONTINUES, "rally_continues")

    def _rally_hit(self, obs: PadelObservation) -> RuleDecision:
        if obs.team is None:
            return self._review("hitter_team_missing")
        if self.last_hitter == obs.team:
            return self._award(_opponent(obs.team), "same_team_double_hit")
        self.last_hitter = obs.team
        self.last_hit_side = obs.side
        self.phase = PointPhase.RALLY
        self._reset_rally_sequence()
        self._evidence.append(self._tag(obs))
        return self._decision(DecisionKind.RALLY_CONTINUES, "legal_player_hit")

    def _rally_floor(self, obs: PadelObservation) -> RuleDecision:
        if self.last_hitter is None:
            return self._review("floor_contact_without_known_hitter")
        if self._bounce_count == 0:
            if obs.side == self.last_hit_side:
                return self._award(_opponent(self.last_hitter), "ball_bounced_on_hitter_side")
            self._bounce_side = obs.side
            self._bounce_count = 1
            return self._decision(DecisionKind.RALLY_CONTINUES, "legal_first_bounce")
        if obs.side == self._bounce_side:
            self._bounce_count += 1
            return self._award(self.last_hitter, "second_bounce")
        return self._review("floor_side_changed_without_player_hit")

    def _rally_exit(self, obs: PadelObservation) -> RuleDecision:
        if self.last_hitter is None:
            return self._review("court_exit_without_known_hitter")
        if self._bounce_count == 0:
            return self._award(_opponent(self.last_hitter), "court_exit_before_opponent_bounce")

        through_gate = bool(obs.metadata.get("through_gate"))
        over_side = bool(obs.metadata.get("over_side_wall"))
        if self.out_of_court_play_enabled and (through_gate or over_side):
            self.phase = PointPhase.OUTSIDE_PLAY
            return self._decision(DecisionKind.RALLY_CONTINUES, "recoverable_outside_play")
        return self._award(self.last_hitter, "unrecoverable_exit_after_legal_bounce")

    def _visibility(self, obs: PadelObservation) -> RuleDecision:
        if obs.visibility == BallVisibility.OUTSIDE_COURT_RECOVERABLE:
            if self.out_of_court_play_enabled and self.phase == PointPhase.RALLY:
                self.phase = PointPhase.OUTSIDE_PLAY
            return self._decision(DecisionKind.NONE, "ball_temporarily_outside_court")
        # Occlusion, unknown depth, and leaving one camera's FOV are perception
        # states only.  They never decide a point.
        return self._decision(DecisionKind.NONE, "visibility_change_only")

    def _interference(self, obs: PadelObservation) -> RuleDecision:
        deliberate = bool(obs.metadata.get("deliberate"))
        offender = obs.team
        if deliberate and offender is not None:
            return self._award(_opponent(offender), "deliberate_interference")
        self.phase = PointPhase.IDLE
        return self._decision(DecisionKind.POINT_LET, "involuntary_interference")

    def _manual_point(self, obs: PadelObservation) -> RuleDecision:
        if obs.team is None:
            return self._review("manual_winner_missing")
        return self._award(obs.team, str(obs.metadata.get("reason", "manual")))

    def _award(self, winner: Optional[TeamId], reason: str) -> RuleDecision:
        if winner is None:
            return self._review("winner_unknown:" + reason)
        self.phase = PointPhase.POINT_ENDED
        return self._decision(DecisionKind.POINT_AWARDED, reason, winner=winner)

    def _review(self, reason: str) -> RuleDecision:
        return self._decision(
            DecisionKind.REVIEW_REQUIRED,
            reason,
            requires_confirmation=True,
        )

    def _reset_serve_sequence(self) -> None:
        self._serve_touched_net = False
        self._serve_landed = False
        self._serve_box_valid = False
        self._bounce_side = None
        self._bounce_count = 0

    def _reset_rally_sequence(self) -> None:
        self._bounce_side = None
        self._bounce_count = 0
        self._ball_crossed_after_hit = False

    def _decision(self, kind: DecisionKind, reason: str,
                  winner: Optional[TeamId] = None, confidence: float = 1.0,
                  requires_confirmation: bool = False) -> RuleDecision:
        return RuleDecision(
            kind=kind,
            phase=self.phase,
            reason=reason,
            winner=winner,
            confidence=confidence,
            requires_confirmation=requires_confirmation,
            evidence=tuple(self._evidence),
        )

    @staticmethod
    def _tag(obs: PadelObservation) -> str:
        detail = obs.surface.value if obs.surface else obs.kind.value
        return f"{obs.timestamp:.3f}:{detail}"

