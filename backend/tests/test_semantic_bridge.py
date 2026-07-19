from logic.padel_rules import PadelRulesEngine
from logic.semantic_bridge import SemanticEventBridge
from models.observations import DecisionKind
from models.types import CourtPoint, EventType, MatchEvent, TeamId


def event(kind, frame, y=18, **metadata):
    return MatchEvent(kind, frame / 30, frame, CourtPoint(5, y),
                      confidence=0.9, metadata=metadata)


def process(engine, bridge, match_event, confidence=0.9):
    decisions = []
    for observation in bridge.translate(match_event, TeamId.TEAM_A, confidence):
        decision = engine.process(observation)
        decisions.append(decision)
        if decision.kind == DecisionKind.SERVE_STARTED:
            bridge.point_ready_accepted()
    return decisions


def test_valid_legacy_serve_becomes_ready_strike_and_legal_floor():
    bridge = SemanticEventBridge({"P1": 1, "P3": 2})
    engine = PadelRulesEngine()
    decisions = process(engine, bridge, event(EventType.SERVE, 30))
    assert [item.kind for item in decisions] == [
        DecisionKind.SERVE_STARTED, DecisionKind.NONE, DecisionKind.NONE]
    assert engine.serve_attempt == 1
    assert engine.phase.value == "return_of_serve"


def test_two_fault_events_follow_first_and_second_service_rules():
    bridge = SemanticEventBridge({})
    engine = PadelRulesEngine()
    first = process(engine, bridge, event(EventType.FAULT, 30, detail="wrong_box"))
    second = process(engine, bridge, event(EventType.FAULT, 60, detail="wrong_box"))
    assert first[-1].kind == DecisionKind.SERVICE_FAULT
    assert second[-1].kind == DecisionKind.POINT_AWARDED
    assert second[-1].winner == TeamId.TEAM_B


def test_low_confidence_event_stays_reviewable_and_retries_point_ready():
    bridge = SemanticEventBridge({})
    engine = PadelRulesEngine()
    low = process(engine, bridge, event(EventType.SERVE, 30), confidence=0.4)
    assert all(item.kind == DecisionKind.REVIEW_REQUIRED for item in low)
    assert not bridge.point_started
    high = process(engine, bridge, event(EventType.SERVE, 60), confidence=0.9)
    assert high[0].kind == DecisionKind.SERVE_STARTED


def test_hit_includes_player_team_for_rally_rules():
    bridge = SemanticEventBridge({"P3": 2})
    translated = bridge.translate(
        event(EventType.HIT, 40, player_id="P3", track_id=7),
        TeamId.TEAM_A, 0.9)
    assert translated[0].team == TeamId.TEAM_B
    assert translated[0].player_id == "P3"
