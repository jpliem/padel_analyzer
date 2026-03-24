"""
PadelScoringEngine — handles point, game, set, and match scoring for padel.

Features:
- Standard padel scoring (0/15/30/40/AD)
- Golden point (Punto de Oro) support
- Configurable sets_to_win (best-of-1 or best-of-3)
- Server rotation tracking with team_players config
- Tiebreak at 6-6 with separate serve index
- Let serve counter and per-point reason history
"""
from __future__ import annotations

from typing import Dict, List, Optional

from models.types import PointReason, ServerInfo, TeamId


class PadelScoringEngine:
    # Standard point labels for regular games
    POINT_LABELS = ["0", "15", "30", "40", "AD"]

    def __init__(
        self,
        golden_point: bool = True,
        sets_to_win: int = 2,
        first_server: Optional[ServerInfo] = None,
        team_players: Optional[Dict[TeamId, List[str]]] = None,
    ):
        # ── Core scoring state ──────────────────────────────────────────────
        self.golden_point = golden_point
        self.sets_to_win = sets_to_win

        self.team1_points = 0
        self.team2_points = 0
        self.team1_games = 0
        self.team2_games = 0
        self.team1_sets = 0
        self.team2_sets = 0
        self.game_over = False

        # ── Tiebreak state ─────────────────────────────────────────────────
        self.is_tiebreak = False
        self._tiebreak_points: List[int] = [0, 0]   # [team1_tb_pts, team2_tb_pts]
        self._tiebreak_total_points = 0
        self._tiebreak_serve_idx = 0   # separate from main serve index

        # ── Server rotation ─────────────────────────────────────────────────
        self.team_players = team_players
        self._serve_order: List[str] = []   # [A1, B1, A2, B2]
        self._serve_game_count = 0          # games served since match start
        self.server_history: List[str] = []
        self.current_server: Optional[ServerInfo] = None

        if first_server is not None and team_players is not None:
            self._build_serve_order(first_server, team_players)
            self.current_server = first_server
            self.server_history.append(first_server.player_id)

        # ── Let / reason tracking ───────────────────────────────────────────
        self.let_count = 0
        self.point_history: List[dict] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def get_score_display(self) -> dict:
        if self.is_tiebreak:
            tb1, tb2 = self._tiebreak_points
            score_str = f"{tb1} - {tb2}"
        else:
            t1_pts = self.POINT_LABELS[self.team1_points]
            t2_pts = self.POINT_LABELS[self.team2_points]
            score_str = f"{t1_pts} - {t2_pts}"
        return {
            "score": score_str,
            "games": f"{self.team1_games} - {self.team2_games}",
            "sets": f"{self.team1_sets} - {self.team2_sets}",
        }

    def add_point(self, team_id: int, reason: Optional[PointReason] = None) -> None:
        if self.game_over:
            return
        score_before = self.get_score_display()["score"]
        if self.is_tiebreak:
            self._tiebreak_point(team_id)
        else:
            self._increment_point(team_id)
        self.point_history.append({
            "team": team_id,
            "reason": reason,
            "score_before": score_before,
        })

    def register_let(self) -> None:
        """Record a let serve (no point awarded)."""
        self.let_count += 1

    # ── Internal: regular scoring ────────────────────────────────────────────

    def _increment_point(self, team_id: int) -> None:
        other_team = 2 if team_id == 1 else 1

        p1 = self.team1_points if team_id == 1 else self.team2_points
        p2 = self.team2_points if team_id == 1 else self.team1_points

        # Golden point: at deuce, next point wins
        if self.golden_point and p1 == 3 and p2 == 3:
            self._win_game(team_id)
            return

        if p1 < 3:
            if team_id == 1:
                self.team1_points += 1
            else:
                self.team2_points += 1
        elif p1 == 3:
            if p2 < 3:
                self._win_game(team_id)
            elif p2 == 3:          # Deuce → Advantage
                if team_id == 1:
                    self.team1_points = 4
                else:
                    self.team2_points = 4
            elif p2 == 4:          # Other had Advantage → back to Deuce
                if team_id == 1:
                    self.team2_points = 3
                else:
                    self.team1_points = 3
        elif p1 == 4:              # Had Advantage → win game
            self._win_game(team_id)

    def _win_game(self, team_id: int) -> None:
        if team_id == 1:
            self.team1_games += 1
        else:
            self.team2_games += 1

        self.team1_points = 0
        self.team2_points = 0

        self._rotate_server()
        self._check_set(team_id)

    def _check_set(self, team_id: int) -> None:
        g1, g2 = self.team1_games, self.team2_games

        # Tiebreak condition: 6-6
        if g1 == 6 and g2 == 6:
            self._start_tiebreak()
            return

        set_won = False
        if (g1 >= 6 and g1 - g2 >= 2) or g1 == 7:
            self.team1_sets += 1
            set_won = True
        elif (g2 >= 6 and g2 - g1 >= 2) or g2 == 7:
            self.team2_sets += 1
            set_won = True

        if set_won:
            self.team1_games = 0
            self.team2_games = 0
            if self.team1_sets >= self.sets_to_win or self.team2_sets >= self.sets_to_win:
                self.game_over = True

    # ── Internal: tiebreak scoring ───────────────────────────────────────────

    def _start_tiebreak(self) -> None:
        self.is_tiebreak = True
        self._tiebreak_points = [0, 0]
        self._tiebreak_total_points = 0
        # Capture the serve index at tiebreak start (tiebreak uses its own rotation)
        self._tiebreak_serve_idx = self._serve_game_count

    def _tiebreak_point(self, team_id: int) -> None:
        idx = team_id - 1            # 0 for team1, 1 for team2
        self._tiebreak_points[idx] += 1
        self._tiebreak_total_points += 1

        tb1, tb2 = self._tiebreak_points
        # Rotate server in tiebreak: change after point 1, then every 2 points
        if self._serve_order:
            if self._tiebreak_total_points == 1 or (
                self._tiebreak_total_points > 1
                and (self._tiebreak_total_points - 1) % 2 == 0
            ):
                self._tiebreak_serve_idx += 1
                self._update_server_from_tiebreak_idx()

        # Check tiebreak win: 7+ with 2-point lead
        if (tb1 >= 7 or tb2 >= 7) and abs(tb1 - tb2) >= 2:
            self._win_tiebreak(1 if tb1 > tb2 else 2)

    def _win_tiebreak(self, team_id: int) -> None:
        if team_id == 1:
            self.team1_sets += 1
        else:
            self.team2_sets += 1
        self.team1_games = 0
        self.team2_games = 0
        self._tiebreak_points = [0, 0]
        self.is_tiebreak = False
        if self.team1_sets >= self.sets_to_win or self.team2_sets >= self.sets_to_win:
            self.game_over = True

    # ── Internal: server rotation ────────────────────────────────────────────

    def _build_serve_order(
        self,
        first_server: ServerInfo,
        team_players: Dict[TeamId, List[str]],
    ) -> None:
        """Build the repeating serve order [A1, B1, A2, B2]."""
        a_players = team_players.get(TeamId.TEAM_A, [])
        b_players = team_players.get(TeamId.TEAM_B, [])
        # Determine starting index within TEAM_A based on first_server
        first_pid = first_server.player_id
        if first_server.team_id == TeamId.TEAM_A:
            a_idx = a_players.index(first_pid) if first_pid in a_players else 0
            b_idx = 0
        else:
            b_idx = b_players.index(first_pid) if first_pid in b_players else 0
            a_idx = 0

        # Rotate player lists so first_server leads his/her team list
        a_ordered = a_players[a_idx:] + a_players[:a_idx]
        b_ordered = b_players[b_idx:] + b_players[:b_idx]

        # Interleave: A1, B1, A2, B2, A3, B3…
        max_len = max(len(a_ordered), len(b_ordered))
        self._serve_order = []
        for i in range(max_len):
            if first_server.team_id == TeamId.TEAM_A:
                if i < len(a_ordered):
                    self._serve_order.append(a_ordered[i])
                if i < len(b_ordered):
                    self._serve_order.append(b_ordered[i])
            else:
                if i < len(b_ordered):
                    self._serve_order.append(b_ordered[i])
                if i < len(a_ordered):
                    self._serve_order.append(a_ordered[i])

    def _rotate_server(self) -> None:
        """Advance to the next server in the rotation after a game is won."""
        if not self._serve_order:
            return
        self._serve_game_count += 1
        idx = self._serve_game_count % len(self._serve_order)
        player_id = self._serve_order[idx]
        # Determine team from team_players mapping
        team_id = self._find_team_for_player(player_id)
        self.current_server = ServerInfo(team_id=team_id, player_id=player_id)
        self.server_history.append(player_id)

    def _update_server_from_tiebreak_idx(self) -> None:
        """Update current_server based on tiebreak serve index."""
        if not self._serve_order:
            return
        idx = self._tiebreak_serve_idx % len(self._serve_order)
        player_id = self._serve_order[idx]
        team_id = self._find_team_for_player(player_id)
        self.current_server = ServerInfo(team_id=team_id, player_id=player_id)

    def _find_team_for_player(self, player_id: str) -> TeamId:
        if self.team_players:
            for team, players in self.team_players.items():
                if player_id in players:
                    return team
        return TeamId.TEAM_A


# Example Usage:
# engine = PadelScoringEngine()
# engine.add_point(1)  # Team 1 wins a point
# print(engine.get_score_display())
