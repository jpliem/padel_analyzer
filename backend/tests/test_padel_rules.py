"""Rule scenarios from the FIP point, service, and out-of-court rules."""

from logic.padel_rules import PadelRulesEngine
from models.observations import (
    BallVisibility,
    CourtSurface,
    DecisionKind,
    ObservationKind,
    PadelObservation,
    PointPhase,
)
from models.types import TeamId


def obs(kind, t, *, team=None, side=None, surface=None, visibility=None,
        confidence=1.0, **metadata):
    return PadelObservation(
        kind=kind, timestamp=t, team=team, side=side, surface=surface,
        visibility=visibility, confidence=confidence, metadata=metadata,
    )


def ready(engine, team=TeamId.TEAM_A):
    return engine.process(obs(ObservationKind.POINT_READY, 0, team=team, side="near"))


def serve_floor(engine, t=1, correct=True):
    return engine.process(obs(
        ObservationKind.SURFACE_CONTACT, t, side="far",
        surface=CourtSurface.FLOOR, correct_service_box=correct,
    ))


def enter_rally(engine):
    ready(engine)
    serve_floor(engine)
    return engine.process(obs(
        ObservationKind.PLAYER_HIT, 1.2, team=TeamId.TEAM_B, side="far",
    ))


def test_first_fault_gives_second_service():
    engine = PadelRulesEngine()
    ready(engine)
    decision = serve_floor(engine, correct=False)
    assert decision.kind == DecisionKind.SERVICE_FAULT
    assert decision.phase == PointPhase.SECOND_SERVE
    assert engine.serve_attempt == 2


def test_two_faults_award_receiver_point():
    engine = PadelRulesEngine()
    ready(engine)
    serve_floor(engine, correct=False)
    decision = serve_floor(engine, t=2, correct=False)
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_B
    assert decision.reason.startswith("double_fault")


def test_net_on_first_service_repeats_first_service():
    engine = PadelRulesEngine()
    ready(engine)
    engine.process(obs(ObservationKind.SURFACE_CONTACT, .5, surface=CourtSurface.NET))
    serve_floor(engine)
    decision = engine.process(obs(
        ObservationKind.PLAYER_HIT, 1.1, team=TeamId.TEAM_B, side="far"))
    assert decision.kind == DecisionKind.SERVICE_LET
    assert decision.phase == PointPhase.FIRST_SERVE
    assert engine.serve_attempt == 1


def test_net_on_second_service_repeats_only_second_service():
    engine = PadelRulesEngine()
    ready(engine)
    serve_floor(engine, correct=False)
    engine.process(obs(ObservationKind.SURFACE_CONTACT, 1.5, surface=CourtSurface.NET))
    serve_floor(engine, t=2)
    decision = engine.process(obs(
        ObservationKind.PLAYER_HIT, 2.1, team=TeamId.TEAM_B, side="far"))
    assert decision.kind == DecisionKind.SERVICE_LET
    assert decision.phase == PointPhase.SECOND_SERVE
    assert engine.serve_attempt == 2


def test_net_then_legal_box_then_fence_is_fault_not_let():
    engine = PadelRulesEngine()
    ready(engine)
    engine.process(obs(ObservationKind.SURFACE_CONTACT, .5, surface=CourtSurface.NET))
    serve_floor(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 1.1, side="far", surface=CourtSurface.FENCE))
    assert decision.kind == DecisionKind.SERVICE_FAULT
    assert decision.phase == PointPhase.SECOND_SERVE


def test_legal_serve_then_glass_remains_returnable():
    engine = PadelRulesEngine()
    ready(engine)
    serve_floor(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 1.1, side="far", surface=CourtSurface.GLASS))
    assert decision.kind == DecisionKind.NONE
    assert decision.phase == PointPhase.RETURN_OF_SERVE


def test_receiver_must_not_volley_serve():
    engine = PadelRulesEngine()
    ready(engine)
    decision = engine.process(obs(
        ObservationKind.PLAYER_HIT, .5, team=TeamId.TEAM_B, side="far"))
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_A


def test_receiver_loses_after_second_service_bounce():
    engine = PadelRulesEngine()
    ready(engine)
    serve_floor(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 1.5, side="far", surface=CourtSurface.FLOOR))
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_A


def test_legal_return_enters_rally():
    engine = PadelRulesEngine()
    decision = enter_rally(engine)
    assert decision.kind == DecisionKind.VALID_SERVE
    assert decision.phase == PointPhase.RALLY


def test_rally_net_touch_is_legal():
    engine = PadelRulesEngine()
    enter_rally(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 2, surface=CourtSurface.NET))
    assert decision.kind == DecisionKind.RALLY_CONTINUES
    assert decision.phase == PointPhase.RALLY


def test_opponent_floor_then_glass_is_legal():
    engine = PadelRulesEngine()
    enter_rally(engine)  # B last hit from far
    engine.process(obs(ObservationKind.SURFACE_CONTACT, 2, side="near", surface=CourtSurface.FLOOR))
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 2.1, side="near", surface=CourtSurface.GLASS))
    assert decision.kind == DecisionKind.RALLY_CONTINUES


def test_opponent_wall_before_floor_loses_point():
    engine = PadelRulesEngine()
    enter_rally(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 2, side="near", surface=CourtSurface.GLASS))
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_A


def test_player_may_use_own_glass_before_crossing():
    engine = PadelRulesEngine()
    enter_rally(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 2, side="far", surface=CourtSurface.GLASS))
    assert decision.kind == DecisionKind.RALLY_CONTINUES


def test_player_may_not_use_own_fence_before_crossing():
    engine = PadelRulesEngine()
    enter_rally(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 2, side="far", surface=CourtSurface.FENCE))
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_A


def test_second_bounce_awards_last_hitter():
    engine = PadelRulesEngine()
    enter_rally(engine)
    engine.process(obs(ObservationKind.SURFACE_CONTACT, 2, side="near", surface=CourtSurface.FLOOR))
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 2.5, side="near", surface=CourtSurface.FLOOR))
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_B


def test_authorized_gate_exit_stays_live():
    engine = PadelRulesEngine(out_of_court_play_enabled=True)
    enter_rally(engine)
    engine.process(obs(ObservationKind.SURFACE_CONTACT, 2, side="near", surface=CourtSurface.FLOOR))
    decision = engine.process(obs(
        ObservationKind.BALL_EXITED, 2.2, side="near", through_gate=True))
    assert decision.kind == DecisionKind.RALLY_CONTINUES
    assert decision.phase == PointPhase.OUTSIDE_PLAY


def test_outside_player_can_return_ball():
    engine = PadelRulesEngine(out_of_court_play_enabled=True)
    enter_rally(engine)
    engine.process(obs(ObservationKind.SURFACE_CONTACT, 2, side="near", surface=CourtSurface.FLOOR))
    engine.process(obs(ObservationKind.BALL_EXITED, 2.2, side="near", through_gate=True))
    decision = engine.process(obs(
        ObservationKind.PLAYER_HIT, 2.5, team=TeamId.TEAM_A, side="near", outside_court=True))
    assert decision.kind == DecisionKind.RALLY_CONTINUES
    assert decision.phase == PointPhase.RALLY


def test_exit_is_point_when_outside_play_not_authorized():
    engine = PadelRulesEngine(out_of_court_play_enabled=False)
    enter_rally(engine)
    engine.process(obs(ObservationKind.SURFACE_CONTACT, 2, side="near", surface=CourtSurface.FLOOR))
    decision = engine.process(obs(
        ObservationKind.BALL_EXITED, 2.2, side="near", through_gate=True))
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_B


def test_camera_visibility_loss_never_ends_point():
    engine = PadelRulesEngine()
    enter_rally(engine)
    for visibility in (BallVisibility.OCCLUDED, BallVisibility.OUTSIDE_FOV,
                       BallVisibility.UNKNOWN):
        decision = engine.process(obs(
            ObservationKind.VISIBILITY_CHANGED, 2, visibility=visibility))
        assert decision.kind == DecisionKind.NONE
        assert decision.phase == PointPhase.RALLY


def test_low_confidence_observation_requires_review_without_state_change():
    engine = PadelRulesEngine(review_threshold=.7)
    enter_rally(engine)
    decision = engine.process(obs(
        ObservationKind.SURFACE_CONTACT, 2, side="near", surface=CourtSurface.FLOOR,
        confidence=.4))
    assert decision.kind == DecisionKind.REVIEW_REQUIRED
    assert decision.requires_confirmation
    assert decision.phase == PointPhase.RALLY


def test_involuntary_interference_replays_point():
    engine = PadelRulesEngine()
    enter_rally(engine)
    decision = engine.process(obs(ObservationKind.INTERFERENCE, 2, deliberate=False))
    assert decision.kind == DecisionKind.POINT_LET
    assert decision.phase == PointPhase.IDLE


def test_deliberate_interference_awards_opponent():
    engine = PadelRulesEngine()
    enter_rally(engine)
    decision = engine.process(obs(
        ObservationKind.INTERFERENCE, 2, team=TeamId.TEAM_A, deliberate=True))
    assert decision.kind == DecisionKind.POINT_AWARDED
    assert decision.winner == TeamId.TEAM_B

