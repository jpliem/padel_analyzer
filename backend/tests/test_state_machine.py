import pytest
from models.types import MatchState, PointReason


class TestMatchStateMachine:
    def test_initial_state_idle(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        assert sm.state == MatchState.IDLE

    def test_serve_transitions_to_serving(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        assert sm.state == MatchState.SERVING_1ST

    def test_valid_serve_transitions_to_rally(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_serve_result(valid=True)
        assert sm.state == MatchState.RALLY

    def test_fault_transitions_to_serving_2nd(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_serve_result(valid=False)
        assert sm.state == MatchState.SERVING_2ND

    def test_double_fault_transitions_to_point_ended(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_serve_result(valid=False)
        sm.on_serve_result(valid=False)
        assert sm.state == MatchState.POINT_ENDED
        assert sm.point_end_reason == PointReason.DOUBLE_FAULT

    def test_rally_point_end(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_serve_result(valid=True)
        sm.on_point_ended(PointReason.DOUBLE_BOUNCE)
        assert sm.state == MatchState.POINT_ENDED
        assert sm.point_end_reason == PointReason.DOUBLE_BOUNCE

    def test_score_update_back_to_idle(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_serve_result(valid=True)
        sm.on_point_ended(PointReason.WINNER)
        sm.on_score_updated()
        assert sm.state == MatchState.IDLE

    def test_let_stays_in_serving(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_let()
        assert sm.state == MatchState.SERVING_1ST

    def test_let_on_2nd_serve(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_serve_result(valid=False)
        sm.on_let()
        assert sm.state == MatchState.SERVING_2ND
