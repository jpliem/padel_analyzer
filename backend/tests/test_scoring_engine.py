"""
Tests for PadelScoringEngine — TDD, built incrementally per task 2A–2E.
"""
import pytest
from logic.scoring_engine import PadelScoringEngine
from models.types import TeamId, ServerInfo, PointReason


# ---------------------------------------------------------------------------
# 2A: Basic Scoring (existing behaviour)
# ---------------------------------------------------------------------------

class TestBasicScoring:
    def test_initial_score(self):
        engine = PadelScoringEngine()
        display = engine.get_score_display()
        assert display["score"] == "0 - 0"
        assert display["games"] == "0 - 0"
        assert display["sets"] == "0 - 0"

    def test_single_point_team1(self):
        engine = PadelScoringEngine()
        engine.add_point(1)
        assert engine.get_score_display()["score"] == "15 - 0"

    def test_point_sequence_to_game(self):
        engine = PadelScoringEngine()
        for _ in range(4):
            engine.add_point(1)
        assert engine.get_score_display()["score"] == "0 - 0"
        assert engine.get_score_display()["games"] == "1 - 0"

    def test_deuce_golden_point(self):
        engine = PadelScoringEngine(golden_point=True)
        # Bring both to 40-40
        for _ in range(3):
            engine.add_point(1)
        for _ in range(3):
            engine.add_point(2)
        assert engine.get_score_display()["score"] == "40 - 40"
        # Next point wins game (golden point)
        engine.add_point(1)
        assert engine.get_score_display()["games"] == "1 - 0"

    def test_deuce_advantage(self):
        engine = PadelScoringEngine(golden_point=False)
        for _ in range(3):
            engine.add_point(1)
        for _ in range(3):
            engine.add_point(2)
        # Deuce → advantage team1
        engine.add_point(1)
        assert engine.get_score_display()["score"] == "AD - 40"
        # Back to deuce
        engine.add_point(2)
        assert engine.get_score_display()["score"] == "40 - 40"
        # Advantage team2
        engine.add_point(2)
        assert engine.get_score_display()["score"] == "40 - AD"
        # Team2 wins game
        engine.add_point(2)
        assert engine.get_score_display()["games"] == "0 - 1"

    def test_set_win_at_6_0(self):
        engine = PadelScoringEngine()
        for _ in range(24):   # 6 games × 4 points each
            engine.add_point(1)
        assert engine.get_score_display()["sets"] == "1 - 0"
        assert engine.get_score_display()["games"] == "0 - 0"

    def test_game_over_after_2_sets(self):
        engine = PadelScoringEngine()
        for _ in range(48):   # 2 sets × 6 games × 4 points
            engine.add_point(1)
        assert engine.game_over is True
        assert engine.get_score_display()["sets"] == "2 - 0"

    def test_no_points_after_game_over(self):
        engine = PadelScoringEngine()
        for _ in range(48):
            engine.add_point(1)
        assert engine.game_over is True
        engine.add_point(2)
        assert engine.get_score_display()["score"] == "0 - 0"


# ---------------------------------------------------------------------------
# 2B: Configurable sets_to_win
# ---------------------------------------------------------------------------

class TestMatchFormat:
    def test_best_of_1(self):
        engine = PadelScoringEngine(sets_to_win=1)
        for _ in range(24):   # 6 games
            engine.add_point(1)
        assert engine.game_over is True
        assert engine.get_score_display()["sets"] == "1 - 0"

    def test_best_of_3_default(self):
        engine = PadelScoringEngine()   # default sets_to_win=2
        for _ in range(24):   # 1 set
            engine.add_point(1)
        assert engine.game_over is False

    def test_best_of_3_requires_2_sets(self):
        engine = PadelScoringEngine(sets_to_win=2)
        for _ in range(48):   # 2 sets
            engine.add_point(1)
        assert engine.game_over is True
        assert engine.get_score_display()["sets"] == "2 - 0"


# ---------------------------------------------------------------------------
# 2C: Server tracking
# ---------------------------------------------------------------------------

class TestServerTracking:
    def test_initial_server(self):
        server = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
        engine = PadelScoringEngine(
            first_server=server,
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]}
        )
        assert engine.current_server.team_id == TeamId.TEAM_A
        assert engine.current_server.player_id == "P1"

    def test_server_rotates_after_game(self):
        server = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
        engine = PadelScoringEngine(
            first_server=server,
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]}
        )
        for _ in range(4):
            engine.add_point(1)
        assert engine.current_server.team_id == TeamId.TEAM_B
        assert engine.current_server.player_id == "P3"

    def test_server_alternates_within_team(self):
        server = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
        engine = PadelScoringEngine(
            first_server=server,
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]}
        )
        # 4 games → 4 rotations; history has initial + 4 = 5 entries
        for game in range(4):
            for _ in range(4):
                engine.add_point(1)
        expected_servers = ["P1", "P3", "P2", "P4", "P1"]
        assert engine.server_history == expected_servers

    def test_server_no_rotation_without_config(self):
        engine = PadelScoringEngine()
        assert engine.current_server is None


# ---------------------------------------------------------------------------
# 2D: Tiebreak
# ---------------------------------------------------------------------------

class TestTiebreak:
    def _play_to_6_6(self, engine):
        """Drive both teams to 6 games each without triggering set win.

        Alternates game wins: T1, T2, T1, T2 … giving 6-5 then 6-6.
        This avoids triggering a set win (which requires >=6 with 2-game lead).
        """
        for i in range(12):
            team = 1 if i % 2 == 0 else 2
            for _ in range(4):
                engine.add_point(team)

    def test_tiebreak_triggers_at_6_6(self):
        engine = PadelScoringEngine()
        self._play_to_6_6(engine)
        assert engine.is_tiebreak is True

    def test_tiebreak_scoring_uses_numbers(self):
        engine = PadelScoringEngine()
        self._play_to_6_6(engine)
        engine.add_point(1)
        assert engine.get_score_display()["score"] == "1 - 0"

    def test_tiebreak_win_at_7_with_2_lead(self):
        engine = PadelScoringEngine()
        self._play_to_6_6(engine)
        for _ in range(7):
            engine.add_point(1)
        assert engine.get_score_display()["sets"] == "1 - 0"

    def test_tiebreak_must_win_by_2(self):
        engine = PadelScoringEngine()
        self._play_to_6_6(engine)
        # Drive to 6-6 in tiebreak
        for _ in range(6):
            engine.add_point(1)
        for _ in range(6):
            engine.add_point(2)
        assert engine.is_tiebreak is True   # still in tiebreak
        # Team1 leads 7-6 — not won yet
        engine.add_point(1)
        assert engine.is_tiebreak is True
        # Team1 leads 8-6 — wins
        engine.add_point(1)
        assert engine.is_tiebreak is False
        assert engine.get_score_display()["sets"] == "1 - 0"

    def test_tiebreak_server_changes_every_2_points(self):
        srv = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
        engine = PadelScoringEngine(
            first_server=srv,
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]}
        )
        self._play_to_6_6(engine)
        # After 6-6, a tiebreak starts. Server for the tiebreak should be
        # whoever's turn it is in the rotation.
        initial_tb_server = engine.current_server.player_id
        # Point 1: server changes after first point
        engine.add_point(1)
        assert engine.current_server.player_id != initial_tb_server
        server_after_1 = engine.current_server.player_id
        # Point 2: no change (still serving)
        engine.add_point(1)
        assert engine.current_server.player_id == server_after_1
        # Point 3: changes again (every 2 points after first)
        engine.add_point(1)
        assert engine.current_server.player_id != server_after_1


# ---------------------------------------------------------------------------
# 2E: Let serve & point reason tracking
# ---------------------------------------------------------------------------

class TestLetServeAndReason:
    def test_register_let(self):
        engine = PadelScoringEngine()
        engine.register_let()
        assert engine.let_count == 1
        assert engine.get_score_display()["score"] == "0 - 0"

    def test_add_point_with_reason(self):
        engine = PadelScoringEngine()
        engine.add_point(1, reason=PointReason.WINNER)
        assert engine.point_history[-1] == {
            "team": 1,
            "reason": PointReason.WINNER,
            "score_before": "0 - 0",
        }

    def test_point_history_tracks_all_points(self):
        engine = PadelScoringEngine()
        engine.add_point(1, reason=PointReason.WINNER)
        engine.add_point(2, reason=PointReason.DOUBLE_FAULT)
        engine.add_point(1, reason=PointReason.OUT)
        assert len(engine.point_history) == 3
