# Phase 1: Foundation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the prototype into a properly structured project with working court calibration (homography), player tracking (ByteTrack), ball detection with Kalman filter, and an upgraded scoring engine — all test-covered.

**Architecture:** Backend Python package with clear separation: `models/` for shared types, `logic/` for scoring, `cv/` for computer vision. All CV modules consume a shared homography matrix from calibration. FastAPI serves the frontend and exposes match setup + calibration endpoints.

**Tech Stack:** Python 3.11+, FastAPI, OpenCV, Ultralytics YOLOv8n, filterpy (Kalman), NumPy, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-padel-analyzer-real-system-design.md`

---

## File Structure

### New Files
```
backend/
  src/
    __init__.py
    models/
      __init__.py
      types.py                    # Shared enums, dataclasses, type definitions
    logic/
      __init__.py                 # (exists but empty — create)
    cv/
      __init__.py                 # (exists but empty — create)
      court_calibration.py        # NEW: Homography-based calibration (replaces calibration.py)
      ball_tracker.py             # NEW: Ball detection + Kalman filter
      player_tracker.py           # NEW: YOLOv8n + ByteTrack player tracking
  tests/
    __init__.py
    conftest.py                   # Shared fixtures
    test_types.py
    test_scoring_engine.py
    test_court_calibration.py
    test_ball_tracker.py
    test_player_tracker.py
    test_api.py
  pytest.ini
```

### Modified Files
```
backend/
  src/
    logic/
      scoring_engine.py           # MODIFY: add tiebreak, server tracking, match format config, let serve
    cv/
      calibration.py              # DEPRECATE: replaced by court_calibration.py
      padel_cv.py                 # DEPRECATE: split into ball_tracker.py + player_tracker.py
      video_processor.py          # DEFERRED: update to new modules in Phase 2
  main.py                         # MODIFY: add match setup, calibration, CORS endpoints
  requirements.txt                # MODIFY: add pytest
```

### Unchanged Files
```
frontend/                         # No frontend changes in Phase 1
demo_analyzer.py                  # Keep as reference
run_real_analysis.py              # Keep as reference
```

---

## Task 1: Project Structure & Shared Types

**Files:**
- Create: `backend/src/__init__.py`
- Create: `backend/src/models/__init__.py`
- Create: `backend/src/models/types.py`
- Create: `backend/src/logic/__init__.py`
- Create: `backend/src/cv/__init__.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_types.py`
- Create: `backend/pytest.ini`
- Modify: `backend/requirements.txt`

- [ ] **Step 1: Create package init files**

Create empty `__init__.py` files to make proper Python packages:

```python
# backend/src/__init__.py
# backend/src/models/__init__.py
# backend/src/logic/__init__.py
# backend/src/cv/__init__.py
# backend/tests/__init__.py
```

All empty files — just package markers.

- [ ] **Step 2: Create pytest config**

```ini
# backend/pytest.ini
[pytest]
testpaths = tests
pythonpath = src
```

- [ ] **Step 3: Update requirements.txt**

Add to `backend/requirements.txt`:

```
# Testing
pytest
pytest-asyncio
httpx  # For FastAPI test client

# Already present: filterpy, scipy, numpy, opencv-python, ultralytics, fastapi, uvicorn
```

- [ ] **Step 4: Install dependencies**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer && source venv/bin/activate && pip install pytest pytest-asyncio httpx`

- [ ] **Step 5: Write the test for shared types**

```python
# backend/tests/test_types.py
from models.types import (
    TeamId, PointReason, MatchFormat, ServerInfo,
    MatchConfig, CourtPoint, BallPosition, PlayerPosition,
    EventType, MatchEvent
)


def test_team_id_values():
    assert TeamId.TEAM_A.value == 1
    assert TeamId.TEAM_B.value == 2


def test_point_reason_values():
    assert PointReason.WINNER.value == "winner"
    assert PointReason.DOUBLE_FAULT.value == "double_fault"
    assert PointReason.OUT.value == "out"
    assert PointReason.NET.value == "net"
    assert PointReason.DOUBLE_BOUNCE.value == "double_bounce"


def test_match_format_values():
    assert MatchFormat.BEST_OF_3.value == 2  # sets to win
    assert MatchFormat.BEST_OF_1.value == 1


def test_server_info():
    s = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
    assert s.team_id == TeamId.TEAM_A
    assert s.player_id == "P1"


def test_match_config_defaults():
    cfg = MatchConfig(
        match_name="Test Match",
        players={"P1": "Alice", "P2": "Bob", "P3": "Carol", "P4": "Dave"},
        teams={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]},
    )
    assert cfg.golden_point is True
    assert cfg.format == MatchFormat.BEST_OF_3
    assert cfg.first_server == ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")


def test_court_point():
    p = CourtPoint(x=5.0, y=10.0)
    assert p.x == 5.0
    assert p.y == 10.0


def test_ball_position():
    bp = BallPosition(x=5.0, y=10.0, z=1.2, speed=45.0, timestamp=1.5)
    assert bp.z == 1.2
    assert bp.speed == 45.0


def test_event_type_values():
    assert EventType.BOUNCE.value == "BOUNCE"
    assert EventType.SERVE.value == "SERVE"
    assert EventType.FAULT.value == "FAULT"
    assert EventType.LET.value == "LET"
    assert EventType.NET_HIT.value == "NET_HIT"
    assert EventType.WALL_HIT.value == "WALL_HIT"
    assert EventType.OUT.value == "OUT"
    assert EventType.POINT_END.value == "POINT_END"


def test_match_event():
    evt = MatchEvent(
        event_type=EventType.BOUNCE,
        timestamp=12.5,
        frame_number=375,
        position=CourtPoint(x=3.0, y=7.0),
        confidence=0.85,
    )
    assert evt.event_type == EventType.BOUNCE
    assert evt.position.x == 3.0
    assert evt.metadata == {}
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && source ../venv/bin/activate && python -m pytest tests/test_types.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'models.types'`

- [ ] **Step 7: Write the types module**

```python
# backend/src/models/types.py
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
    BEST_OF_3 = 2  # sets to win
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
```

- [ ] **Step 8: Run test to verify it passes**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_types.py -v`

Expected: All 10 tests PASS

- [ ] **Step 9: Create conftest.py with shared fixtures**

```python
# backend/tests/conftest.py
import pytest
import numpy as np
from models.types import MatchConfig, MatchFormat, ServerInfo, TeamId


@pytest.fixture
def default_match_config():
    return MatchConfig()


@pytest.fixture
def golden_point_config():
    return MatchConfig(golden_point=True)


@pytest.fixture
def advantage_config():
    return MatchConfig(golden_point=False)


@pytest.fixture
def best_of_1_config():
    return MatchConfig(format=MatchFormat.BEST_OF_1)


@pytest.fixture
def sample_court_corners_pixels():
    """4 court corners as they might appear in a 1080p frame (behind-baseline camera).
    Near baseline (y=0 in court) = bottom of frame (large pixel y).
    Far baseline (y=20 in court) = top of frame (small pixel y)."""
    return np.array([
        [320, 700],   # near-left (bottom-left of frame)
        [1600, 700],  # near-right (bottom-right of frame)
        [1200, 200],  # far-right (top-right, narrower due to perspective)
        [720, 200],   # far-left (top-left, narrower due to perspective)
    ], dtype=np.float32)


@pytest.fixture
def court_real_coords():
    """Real-world court corners in meters (10x20m padel court)."""
    return np.array([
        [0, 0],    # near-left
        [10, 0],   # near-right
        [10, 20],  # far-right
        [0, 20],   # far-left
    ], dtype=np.float32)
```

- [ ] **Step 10: Commit**

```bash
git add backend/src/__init__.py backend/src/models/ backend/src/logic/__init__.py backend/src/cv/__init__.py backend/tests/ backend/pytest.ini backend/requirements.txt
git commit -m "feat: add project structure, shared types, and test framework"
```

---

## Task 2: Scoring Engine Upgrades (TDD)

**Files:**
- Create: `backend/tests/test_scoring_engine.py`
- Modify: `backend/src/logic/scoring_engine.py`

### 2A: Test existing behavior

- [ ] **Step 1: Write tests for current scoring behavior**

```python
# backend/tests/test_scoring_engine.py
import pytest
from logic.scoring_engine import PadelScoringEngine


class TestBasicScoring:
    def test_initial_score(self):
        engine = PadelScoringEngine()
        score = engine.get_score_display()
        assert score["score"] == "0 - 0"
        assert score["games"] == "0 - 0"
        assert score["sets"] == "0 - 0"

    def test_single_point_team1(self):
        engine = PadelScoringEngine()
        engine.add_point(1)
        assert engine.get_score_display()["score"] == "15 - 0"

    def test_point_sequence_to_game(self):
        engine = PadelScoringEngine()
        for _ in range(4):
            engine.add_point(1)
        assert engine.get_score_display()["games"] == "1 - 0"
        assert engine.get_score_display()["score"] == "0 - 0"

    def test_deuce_golden_point(self):
        engine = PadelScoringEngine(golden_point=True)
        # Get to 40-40
        for _ in range(3):
            engine.add_point(1)
            engine.add_point(2)
        # Golden point: next point wins
        engine.add_point(1)
        assert engine.get_score_display()["games"] == "1 - 0"

    def test_deuce_advantage(self):
        engine = PadelScoringEngine(golden_point=False)
        # Get to 40-40
        for _ in range(3):
            engine.add_point(1)
            engine.add_point(2)
        # Advantage
        engine.add_point(1)
        assert engine.get_score_display()["score"] == "AD - 40"
        # Back to deuce
        engine.add_point(2)
        assert engine.get_score_display()["score"] == "40 - 40"
        # Advantage + win
        engine.add_point(2)
        engine.add_point(2)
        assert engine.get_score_display()["games"] == "0 - 1"

    def test_set_win_at_6_0(self):
        engine = PadelScoringEngine()
        for _ in range(6):
            for _ in range(4):
                engine.add_point(1)
        assert engine.get_score_display()["sets"] == "1 - 0"

    def test_game_over_after_2_sets(self):
        engine = PadelScoringEngine()
        # Win 2 sets (6-0, 6-0)
        for _ in range(12):
            for _ in range(4):
                engine.add_point(1)
        assert engine.game_over is True

    def test_no_points_after_game_over(self):
        engine = PadelScoringEngine()
        for _ in range(12):
            for _ in range(4):
                engine.add_point(1)
        engine.add_point(2)  # should be ignored
        assert engine.get_score_display()["score"] == "0 - 0"
```

- [ ] **Step 2: Run tests to verify existing code passes**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py -v`

Expected: All 8 tests PASS (these test the current implementation)

### 2B: Match format config

- [ ] **Step 3: Write test for configurable sets_to_win**

Add to `test_scoring_engine.py`:

```python
class TestMatchFormat:
    def test_best_of_1(self):
        engine = PadelScoringEngine(golden_point=True, sets_to_win=1)
        # Win 1 set (6-0)
        for _ in range(6):
            for _ in range(4):
                engine.add_point(1)
        assert engine.game_over is True
        assert engine.team1_sets == 1

    def test_best_of_3_default(self):
        engine = PadelScoringEngine()
        # Win 1 set — match NOT over
        for _ in range(6):
            for _ in range(4):
                engine.add_point(1)
        assert engine.game_over is False
        assert engine.team1_sets == 1

    def test_best_of_3_requires_2_sets(self):
        engine = PadelScoringEngine(sets_to_win=2)
        for _ in range(12):
            for _ in range(4):
                engine.add_point(1)
        assert engine.game_over is True
        assert engine.team1_sets == 2
```

- [ ] **Step 4: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py::TestMatchFormat -v`

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'sets_to_win'`

- [ ] **Step 5: Implement sets_to_win parameter**

Modify `backend/src/logic/scoring_engine.py` — change `__init__`:

```python
def __init__(self, golden_point=True, sets_to_win=2):
    self.scores = ["0", "15", "30", "40", "AD"]
    self.team1_points = 0
    self.team2_points = 0
    self.team1_games = 0
    self.team2_games = 0
    self.team1_sets = 0
    self.team2_sets = 0
    self.golden_point = golden_point
    self.sets_to_win = sets_to_win
    self.game_over = False
```

Change `_check_set` line 83 from `== 2` to `== self.sets_to_win`:

```python
if self.team1_sets == self.sets_to_win or self.team2_sets == self.sets_to_win:
    self.game_over = True
```

- [ ] **Step 6: Run tests — expect pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py -v`

Expected: All 11 tests PASS

### 2C: Server tracking

- [ ] **Step 7: Write server tracking tests**

Add to `test_scoring_engine.py`:

```python
from models.types import TeamId, ServerInfo


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
        # Win game 1 — server should rotate to team B
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
        # Game 1: P1 serves → game 2: P3 serves → game 3: P2 serves → game 4: P4 serves
        # server_history includes initial + 4 rotations = 5 entries
        for game in range(4):
            for _ in range(4):
                engine.add_point(1)

        expected_servers = ["P1", "P3", "P2", "P4", "P1"]
        assert engine.server_history == expected_servers

    def test_server_no_rotation_without_config(self):
        """Without server config, server tracking is None (backwards compat)."""
        engine = PadelScoringEngine()
        assert engine.current_server is None
```

- [ ] **Step 8: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py::TestServerTracking -v`

Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'first_server'`

- [ ] **Step 9: Implement server tracking**

Update `backend/src/logic/scoring_engine.py`:

```python
from models.types import TeamId, ServerInfo


class PadelScoringEngine:
    def __init__(self, golden_point=True, sets_to_win=2,
                 first_server=None, team_players=None):
        self.scores = ["0", "15", "30", "40", "AD"]
        self.team1_points = 0
        self.team2_points = 0
        self.team1_games = 0
        self.team2_games = 0
        self.team1_sets = 0
        self.team2_sets = 0
        self.golden_point = golden_point
        self.sets_to_win = sets_to_win
        self.game_over = False

        # Server tracking
        self.current_server = first_server
        self.team_players = team_players or {}
        self.server_history = []
        self._serve_order = []
        if first_server and team_players:
            self._build_serve_order(first_server, team_players)
            self.server_history.append(first_server.player_id)

    def _build_serve_order(self, first_server, team_players):
        """Build repeating serve order: A1, B1, A2, B2, A1, B1..."""
        team_a = team_players.get(TeamId.TEAM_A, [])
        team_b = team_players.get(TeamId.TEAM_B, [])

        if first_server.team_id == TeamId.TEAM_A:
            first_team, second_team = team_a, team_b
            first_tid, second_tid = TeamId.TEAM_A, TeamId.TEAM_B
        else:
            first_team, second_team = team_b, team_a
            first_tid, second_tid = TeamId.TEAM_B, TeamId.TEAM_A

        # Find starting index in first team
        start_idx = first_team.index(first_server.player_id) if first_server.player_id in first_team else 0

        self._serve_order = [
            ServerInfo(team_id=first_tid, player_id=first_team[start_idx % len(first_team)]),
            ServerInfo(team_id=second_tid, player_id=second_team[0]),
            ServerInfo(team_id=first_tid, player_id=first_team[(start_idx + 1) % len(first_team)]),
            ServerInfo(team_id=second_tid, player_id=second_team[1 % len(second_team)]),
        ]
        self._serve_game_count = 0

    def _rotate_server(self):
        if not self._serve_order:
            return
        self._serve_game_count += 1
        idx = self._serve_game_count % len(self._serve_order)
        self.current_server = self._serve_order[idx]
        self.server_history.append(self.current_server.player_id)
```

Then update `_win_game` to call `self._rotate_server()`:

```python
def _win_game(self, team_id):
    if team_id == 1:
        self.team1_games += 1
    else:
        self.team2_games += 1

    self.team1_points = 0
    self.team2_points = 0
    self._rotate_server()
    self._check_set(team_id)
```

- [ ] **Step 10: Run tests — expect pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py -v`

Expected: All 15 tests PASS

### 2D: Tiebreak

- [ ] **Step 11: Write tiebreak tests**

Add to `test_scoring_engine.py`:

```python
class TestTiebreak:
    def _get_to_6_6(self, engine):
        """Helper: play to 6-6 in games."""
        for _ in range(6):
            for _ in range(4):
                engine.add_point(1)
            for _ in range(4):
                engine.add_point(2)
        assert engine.team1_games == 6
        assert engine.team2_games == 6

    def test_tiebreak_triggers_at_6_6(self):
        engine = PadelScoringEngine()
        self._get_to_6_6(engine)
        assert engine.is_tiebreak is True

    def test_tiebreak_scoring_uses_numbers(self):
        engine = PadelScoringEngine()
        self._get_to_6_6(engine)
        engine.add_point(1)
        score = engine.get_score_display()
        assert score["score"] == "1 - 0"

    def test_tiebreak_win_at_7_with_2_lead(self):
        engine = PadelScoringEngine()
        self._get_to_6_6(engine)
        for _ in range(7):
            engine.add_point(1)
        assert engine.team1_sets == 1
        assert engine.team1_games == 0  # reset after set

    def test_tiebreak_must_win_by_2(self):
        engine = PadelScoringEngine()
        self._get_to_6_6(engine)
        # Get to 6-6 in tiebreak
        for _ in range(6):
            engine.add_point(1)
            engine.add_point(2)
        # Now 6-6 in TB — need 2 point lead
        engine.add_point(1)  # 7-6
        assert engine.is_tiebreak is True  # still in tiebreak
        engine.add_point(1)  # 8-6 — wins
        assert engine.team1_sets == 1

    def test_tiebreak_server_changes_every_2_points(self):
        server = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
        engine = PadelScoringEngine(
            first_server=server,
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]}
        )
        self._get_to_6_6(engine)
        # In tiebreak: first server serves 1 point, then alternate every 2
        initial = engine.current_server.player_id
        engine.add_point(1)  # Point 1 — first server
        first = engine.current_server.player_id
        engine.add_point(1)  # Point 2 — server changes
        second = engine.current_server.player_id
        assert first != second  # server changed after point 1 (odd point in TB)
```

- [ ] **Step 12: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py::TestTiebreak -v`

Expected: FAIL — `AttributeError: 'PadelScoringEngine' object has no attribute 'is_tiebreak'`

- [ ] **Step 13: Implement tiebreak logic**

Add to `PadelScoringEngine.__init__`:
```python
self.is_tiebreak = False
self._tiebreak_points = [0, 0]  # [team1, team2]
self._tiebreak_total_points = 0
self._tiebreak_serve_idx = 0  # separate index for tiebreak server rotation
```

Add new method:
```python
def _start_tiebreak(self):
    self.is_tiebreak = True
    self._tiebreak_points = [0, 0]
    self._tiebreak_total_points = 0
    self._tiebreak_serve_idx = 0
```

Modify `add_point` to handle tiebreak:
```python
def add_point(self, team_id):
    if self.game_over:
        return
    if self.is_tiebreak:
        self._tiebreak_point(team_id)
    else:
        self._increment_point(team_id)

def _tiebreak_point(self, team_id):
    idx = 0 if team_id == 1 else 1
    self._tiebreak_points[idx] += 1
    self._tiebreak_total_points += 1

    p1, p2 = self._tiebreak_points
    # Tiebreak server rotation: first player serves 1 point, then alternate every 2.
    # Rotate after points 1, 3, 5, 7... (using separate tiebreak counter, not _serve_game_count)
    if self._tiebreak_total_points == 1 or (self._tiebreak_total_points > 1 and (self._tiebreak_total_points - 1) % 2 == 0):
        self._tiebreak_serve_idx += 1
        if self._serve_order:
            # Cycle through serve order using tiebreak-specific index
            order_idx = (self._serve_game_count + self._tiebreak_serve_idx) % len(self._serve_order)
            self.current_server = self._serve_order[order_idx]
            self.server_history.append(self.current_server.player_id)

    # Check win: first to 7 with 2-point lead
    if p1 >= 7 and p1 - p2 >= 2:
        self._win_tiebreak(1)
    elif p2 >= 7 and p2 - p1 >= 2:
        self._win_tiebreak(2)

def _win_tiebreak(self, team_id):
    self.is_tiebreak = False
    self._tiebreak_points = [0, 0]
    self._tiebreak_total_points = 0
    if team_id == 1:
        self.team1_games = 7
    else:
        self.team2_games = 7
    # Advance the main serve counter past the tiebreak so next set starts correctly
    self._serve_game_count += 1
    self._check_set(team_id)
```

Modify `get_score_display` for tiebreak:
```python
def get_score_display(self):
    if self.is_tiebreak:
        return {
            "score": f"{self._tiebreak_points[0]} - {self._tiebreak_points[1]}",
            "games": f"{self.team1_games} - {self.team2_games}",
            "sets": f"{self.team1_sets} - {self.team2_sets}",
            "tiebreak": True,
        }
    t1_pts = self.scores[self.team1_points]
    t2_pts = self.scores[self.team2_points]
    return {
        "score": f"{t1_pts} - {t2_pts}",
        "games": f"{self.team1_games} - {self.team2_games}",
        "sets": f"{self.team1_sets} - {self.team2_sets}",
    }
```

Modify `_check_set` to trigger tiebreak at 6-6 instead of letting games go to 7:
```python
def _check_set(self, team_id):
    g1, g2 = self.team1_games, self.team2_games

    if g1 == 6 and g2 == 6:
        self._start_tiebreak()
        return

    if (g1 >= 6 and g1 - g2 >= 2) or g1 == 7:
        self.team1_sets += 1
        self.team1_games = 0
        self.team2_games = 0
    elif (g2 >= 6 and g2 - g1 >= 2) or g2 == 7:
        self.team2_sets += 1
        self.team1_games = 0
        self.team2_games = 0

    if self.team1_sets == self.sets_to_win or self.team2_sets == self.sets_to_win:
        self.game_over = True
```

- [ ] **Step 14: Run tests — expect pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py -v`

Expected: All 21 tests PASS

### 2E: Let serve & point reason tracking

- [ ] **Step 15: Write let serve and point reason tests**

Add to `test_scoring_engine.py`:

```python
class TestLetServeAndReason:
    def test_register_let(self):
        engine = PadelScoringEngine()
        engine.register_let()
        assert engine.let_count == 1
        # Score unchanged
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
```

Add import at top: `from models.types import TeamId, ServerInfo, PointReason`

- [ ] **Step 16: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py::TestLetServeAndReason -v`

Expected: FAIL

- [ ] **Step 17: Implement let serve and point reason tracking**

Add to `__init__`:
```python
self.let_count = 0
self.point_history = []
```

Add method:
```python
def register_let(self):
    self.let_count += 1
```

Modify `add_point` — combine the tiebreak dispatch (from Step 13) with reason tracking:
```python
def add_point(self, team_id, reason=None):
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
```

Note: This is the final version of `add_point` combining tiebreak dispatch (Step 13) + reason tracking. The intermediate version in Step 13 should use this same signature.

- [ ] **Step 18: Run all scoring tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_scoring_engine.py -v`

Expected: All 23 tests PASS

- [ ] **Step 19: Commit**

```bash
git add backend/src/logic/scoring_engine.py backend/tests/test_scoring_engine.py
git commit -m "feat: upgrade scoring engine — tiebreak, server tracking, match format config, let serve, point history"
```

---

## Task 3: Court Calibration (Homography)

**Files:**
- Create: `backend/src/cv/court_calibration.py`
- Create: `backend/tests/test_court_calibration.py`

- [ ] **Step 1: Write calibration tests**

```python
# backend/tests/test_court_calibration.py
import numpy as np
import pytest
from cv.court_calibration import CourtCalibration

# Padel court: 10m wide (x), 20m long (y)
# Net at y=10m
# Service boxes: y=6.95 to y=13.05, split at x=5


class TestHomographyCalibration:
    def test_calibrate_from_4_corners(self, sample_court_corners_pixels, court_real_coords):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.homography is not None
        assert cal.homography.shape == (3, 3)

    def test_pixel_to_court_at_corners(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        # Near-left corner pixel (320, 700) should map to (0, 0)
        result = cal.pixel_to_court(320, 700)
        assert abs(result[0] - 0.0) < 0.1
        assert abs(result[1] - 0.0) < 0.1

    def test_pixel_to_court_far_right(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        # Far-right pixel (1200, 200) should map to (10, 20)
        result = cal.pixel_to_court(1200, 200)
        assert abs(result[0] - 10.0) < 0.1
        assert abs(result[1] - 20.0) < 0.1

    def test_court_to_pixel_roundtrip(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        # Court center (5, 10) → pixel → back to court
        px, py = cal.court_to_pixel(5.0, 10.0)
        rx, ry = cal.pixel_to_court(px, py)
        assert abs(rx - 5.0) < 0.2
        assert abs(ry - 10.0) < 0.2

    def test_raises_without_calibration(self):
        cal = CourtCalibration()
        with pytest.raises(RuntimeError, match="not calibrated"):
            cal.pixel_to_court(100, 100)

    def test_requires_4_points(self):
        cal = CourtCalibration()
        with pytest.raises(ValueError, match="4 corner points"):
            cal.calibrate(np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32))


class TestCourtZones:
    def test_is_in_bounds(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.is_in_bounds(5.0, 10.0) is True   # court center
        assert cal.is_in_bounds(-1.0, 10.0) is False  # off left
        assert cal.is_in_bounds(11.0, 10.0) is False  # off right
        assert cal.is_in_bounds(5.0, -1.0) is False   # off near
        assert cal.is_in_bounds(5.0, 21.0) is False   # off far

    def test_get_court_side(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.get_court_side(5.0, 5.0) == "near"   # y < 10
        assert cal.get_court_side(5.0, 15.0) == "far"   # y > 10

    def test_is_in_service_box(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        # Service box: y=6.95-13.05, left half (x=0-5) or right half (x=5-10)
        assert cal.is_in_service_box(3.0, 8.0, "near_left") is True
        assert cal.is_in_service_box(7.0, 8.0, "near_right") is True
        assert cal.is_in_service_box(3.0, 12.0, "far_left") is True
        assert cal.is_in_service_box(3.0, 5.0, "near_left") is False  # outside service box

    def test_net_line_y(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.NET_Y == 10.0


class TestCalibrationPersistence:
    def test_to_dict_and_from_dict(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        data = cal.to_dict()
        assert "homography" in data
        assert "corners_pixels" in data

        cal2 = CourtCalibration.from_dict(data)
        result = cal2.pixel_to_court(320, 700)
        assert abs(result[0] - 0.0) < 0.1
```

- [ ] **Step 2: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_court_calibration.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'cv.court_calibration'`

- [ ] **Step 3: Implement CourtCalibration**

```python
# backend/src/cv/court_calibration.py
import cv2
import numpy as np
from typing import Tuple, Optional, Dict


# Padel court dimensions (meters)
COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
NET_Y = 10.0
SERVICE_NEAR_Y = 6.95
SERVICE_FAR_Y = 13.05
SERVICE_CENTER_X = 5.0


class CourtCalibration:
    NET_Y = NET_Y

    def __init__(self):
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        self.corners_pixels: Optional[np.ndarray] = None

    def calibrate(self, corners_pixels: np.ndarray) -> None:
        if len(corners_pixels) != 4:
            raise ValueError("Exactly 4 corner points required")

        self.corners_pixels = corners_pixels.astype(np.float32)
        court_corners = np.array([
            [0, 0],
            [COURT_WIDTH, 0],
            [COURT_WIDTH, COURT_LENGTH],
            [0, COURT_LENGTH],
        ], dtype=np.float32)

        self.homography, _ = cv2.findHomography(corners_pixels, court_corners)
        self.inverse_homography, _ = cv2.findHomography(court_corners, corners_pixels)

    def _check_calibrated(self):
        if self.homography is None:
            raise RuntimeError("Court not calibrated — call calibrate() first")

    def pixel_to_court(self, px: float, py: float) -> Tuple[float, float]:
        self._check_calibrated()
        point = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.homography)
        return float(result[0][0][0]), float(result[0][0][1])

    def court_to_pixel(self, cx: float, cy: float) -> Tuple[float, float]:
        self._check_calibrated()
        point = np.array([[[cx, cy]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.inverse_homography)
        return float(result[0][0][0]), float(result[0][0][1])

    def is_in_bounds(self, x: float, y: float) -> bool:
        return 0 <= x <= COURT_WIDTH and 0 <= y <= COURT_LENGTH

    def get_court_side(self, x: float, y: float) -> str:
        return "near" if y < NET_Y else "far"

    def is_in_service_box(self, x: float, y: float, box: str) -> bool:
        boxes = {
            "near_left": (0, SERVICE_CENTER_X, SERVICE_NEAR_Y, NET_Y),
            "near_right": (SERVICE_CENTER_X, COURT_WIDTH, SERVICE_NEAR_Y, NET_Y),
            "far_left": (0, SERVICE_CENTER_X, NET_Y, SERVICE_FAR_Y),
            "far_right": (SERVICE_CENTER_X, COURT_WIDTH, NET_Y, SERVICE_FAR_Y),
        }
        if box not in boxes:
            return False
        x1, x2, y1, y2 = boxes[box]
        return x1 <= x <= x2 and y1 <= y <= y2

    def to_dict(self) -> Dict:
        self._check_calibrated()
        return {
            "homography": self.homography.tolist(),
            "inverse_homography": self.inverse_homography.tolist(),
            "corners_pixels": self.corners_pixels.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "CourtCalibration":
        cal = cls()
        cal.homography = np.array(data["homography"], dtype=np.float64)
        cal.inverse_homography = np.array(data["inverse_homography"], dtype=np.float64)
        cal.corners_pixels = np.array(data["corners_pixels"], dtype=np.float32)
        return cal
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_court_calibration.py -v`

Expected: All 11 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cv/court_calibration.py backend/tests/test_court_calibration.py
git commit -m "feat: homography-based court calibration with zone detection and persistence"
```

---

## Task 4: Ball Tracker (Detection + Kalman Filter)

**Files:**
- Create: `backend/src/cv/ball_tracker.py`
- Create: `backend/tests/test_ball_tracker.py`

- [ ] **Step 1: Write ball tracker tests**

```python
# backend/tests/test_ball_tracker.py
import numpy as np
import pytest
from cv.ball_tracker import BallTracker
from cv.court_calibration import CourtCalibration


@pytest.fixture
def calibrated_court(sample_court_corners_pixels):
    cal = CourtCalibration()
    cal.calibrate(sample_court_corners_pixels)
    return cal


@pytest.fixture
def tracker(calibrated_court):
    return BallTracker(calibrated_court, fps=30)


class TestBallDetection:
    def test_update_with_detection(self, tracker):
        pos = tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert pos is not None
        assert "x" in pos and "y" in pos

    def test_update_without_detection_returns_prediction(self, tracker):
        # Give it a few detections first
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        tracker.update(bbox=[510, 405, 530, 425], frame_number=1)
        tracker.update(bbox=[520, 410, 540, 430], frame_number=2)
        # Now miss a detection
        pos = tracker.update(bbox=None, frame_number=3)
        # Kalman filter should predict
        assert pos is not None

    def test_no_prediction_without_initial_detection(self, tracker):
        pos = tracker.update(bbox=None, frame_number=0)
        assert pos is None

    def test_lost_after_many_misses(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        # Miss many frames
        for i in range(1, 62):
            tracker.update(bbox=None, frame_number=i)
        assert tracker.is_lost is True


class TestTrajectory:
    def test_trajectory_stored(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        tracker.update(bbox=[510, 410, 530, 430], frame_number=1)
        assert len(tracker.trajectory) == 2

    def test_trajectory_has_court_coords(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        pos = tracker.trajectory[0]
        assert "x" in pos and "y" in pos and "timestamp" in pos


class TestSpeedEstimation:
    def test_speed_computed(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        tracker.update(bbox=[600, 400, 620, 420], frame_number=1)
        assert tracker.trajectory[-1]["speed"] >= 0

    def test_speed_zero_on_first_frame(self, tracker):
        tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert tracker.trajectory[0]["speed"] == 0.0


class TestZEstimation:
    def test_z_estimated_from_bbox_size(self, tracker):
        # Larger bbox = closer to camera = higher
        tracker.update(bbox=[500, 400, 540, 440], frame_number=0)  # 40x40 box
        z1 = tracker.trajectory[0]["z"]

        tracker2 = BallTracker(tracker.calibration, fps=30)
        tracker2.update(bbox=[500, 400, 560, 460], frame_number=0)  # 60x60 box — bigger
        z2 = tracker2.trajectory[0]["z"]

        assert z2 > z1  # bigger box → higher z estimate
```

- [ ] **Step 2: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_ball_tracker.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'cv.ball_tracker'`

- [ ] **Step 3: Implement BallTracker**

```python
# backend/src/cv/ball_tracker.py
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional, List, Dict
from cv.court_calibration import CourtCalibration

# Known padel ball diameter ~6.5cm
BALL_DIAMETER_M = 0.065
LOST_THRESHOLD_FRAMES = 60  # ~2 seconds at 30fps


class BallTracker:
    def __init__(self, calibration: CourtCalibration, fps: float = 30.0):
        self.calibration = calibration
        self.fps = fps
        self.dt = 1.0 / fps

        self.trajectory: List[Dict] = []
        self.is_lost = False
        self._miss_count = 0
        self._initialized = False
        self._prev_court_pos = None

        # Kalman filter: state = [x, y, vx, vy], measurement = [x, y]
        self._kf = KalmanFilter(dim_x=4, dim_z=2)
        self._kf.F = np.array([
            [1, 0, self.dt, 0],
            [0, 1, 0, self.dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        self._kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        self._kf.P *= 100
        self._kf.R = np.eye(2) * 5
        self._kf.Q = np.eye(4) * 0.1

        # For Z estimation: expected ball size at ground level (pixels)
        self._ground_ball_size: Optional[float] = None

    def update(self, bbox: Optional[List[float]], frame_number: int) -> Optional[Dict]:
        timestamp = frame_number * self.dt

        if bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            bbox_size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])

            court_x, court_y = self.calibration.pixel_to_court(cx, cy)
            z = self._estimate_z(bbox_size, court_x, court_y)

            if not self._initialized:
                self._kf.x = np.array([court_x, court_y, 0, 0])
                self._initialized = True
            else:
                self._kf.predict()
                self._kf.update(np.array([court_x, court_y]))

            self._miss_count = 0
            self.is_lost = False

            speed = self._compute_speed(court_x, court_y)
            self._prev_court_pos = (court_x, court_y)

            pos = {
                "x": float(self._kf.x[0]),
                "y": float(self._kf.x[1]),
                "z": z,
                "speed": speed,
                "timestamp": timestamp,
                "frame": frame_number,
                "detected": True,
            }
            self.trajectory.append(pos)
            return pos

        else:
            if not self._initialized:
                return None

            self._miss_count += 1
            if self._miss_count >= LOST_THRESHOLD_FRAMES:
                self.is_lost = True
                return None

            self._kf.predict()
            court_x = float(self._kf.x[0])
            court_y = float(self._kf.x[1])
            speed = self._compute_speed(court_x, court_y)
            self._prev_court_pos = (court_x, court_y)

            pos = {
                "x": court_x,
                "y": court_y,
                "z": 0.0,
                "speed": speed,
                "timestamp": timestamp,
                "frame": frame_number,
                "detected": False,
            }
            self.trajectory.append(pos)
            return pos

    def _estimate_z(self, bbox_size: float, court_x: float, court_y: float) -> float:
        if self._ground_ball_size is None:
            self._ground_ball_size = bbox_size
            return 0.0

        if self._ground_ball_size <= 0:
            return 0.0

        ratio = bbox_size / self._ground_ball_size
        z = max(0.0, (ratio - 1.0) * 3.0)
        return round(z, 2)

    def _compute_speed(self, x: float, y: float) -> float:
        if self._prev_court_pos is None:
            return 0.0
        dx = x - self._prev_court_pos[0]
        dy = y - self._prev_court_pos[1]
        dist = np.sqrt(dx**2 + dy**2)
        speed_ms = dist / self.dt
        speed_kmh = speed_ms * 3.6
        return round(speed_kmh, 1)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_ball_tracker.py -v`

Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cv/ball_tracker.py backend/tests/test_ball_tracker.py
git commit -m "feat: ball tracker with Kalman filter, speed estimation, and Z-height estimation"
```

---

## Task 5: Player Tracker (YOLOv8n + ByteTrack)

**Files:**
- Create: `backend/src/cv/player_tracker.py`
- Create: `backend/tests/test_player_tracker.py`

- [ ] **Step 1: Write player tracker tests**

```python
# backend/tests/test_player_tracker.py
import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from cv.player_tracker import PlayerTracker
from cv.court_calibration import CourtCalibration


@pytest.fixture
def calibrated_court(sample_court_corners_pixels):
    cal = CourtCalibration()
    cal.calibrate(sample_court_corners_pixels)
    return cal


@pytest.fixture
def mock_detections():
    """Simulated YOLO detections: list of [x1, y1, x2, y2, conf, cls]."""
    return np.array([
        [100, 200, 160, 400, 0.9, 0],  # Player 1
        [400, 200, 460, 400, 0.85, 0],  # Player 2
        [800, 500, 860, 700, 0.88, 0],  # Player 3
        [1100, 500, 1160, 700, 0.92, 0],  # Player 4
    ])


class TestPlayerDetection:
    def test_detect_players_returns_positions(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = tracker.update(mock_detections, frame_number=0)
        assert len(positions) > 0

    def test_player_positions_have_court_coords(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        positions = tracker.update(mock_detections, frame_number=0)
        for pos in positions:
            assert "x" in pos and "y" in pos
            assert "track_id" in pos

    def test_empty_detections(self, calibrated_court):
        tracker = PlayerTracker(calibrated_court)
        positions = tracker.update(np.array([]).reshape(0, 6), frame_number=0)
        assert len(positions) == 0


class TestPlayerAssignment:
    def test_assign_player_id(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        # Assign track_id 1 → P1
        tracker.assign_player(track_id=1, player_id="P1")
        assert tracker.get_player_id(track_id=1) == "P1"

    def test_unassigned_returns_none(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        assert tracker.get_player_id(track_id=999) is None

    def test_get_player_position(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        tracker.assign_player(track_id=1, player_id="P1")
        pos = tracker.get_player_position("P1")
        assert pos is not None
        assert "x" in pos and "y" in pos


class TestClosestPlayer:
    def test_find_closest_player(self, calibrated_court, mock_detections):
        tracker = PlayerTracker(calibrated_court)
        tracker.update(mock_detections, frame_number=0)
        tracker.assign_player(track_id=1, player_id="P1")
        tracker.assign_player(track_id=2, player_id="P2")

        # Get P1's position and query closest
        p1_pos = tracker.get_player_position("P1")
        closest = tracker.find_closest_player(p1_pos["x"], p1_pos["y"])
        assert closest == "P1"
```

- [ ] **Step 2: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_player_tracker.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'cv.player_tracker'`

- [ ] **Step 3: Implement PlayerTracker**

```python
# backend/src/cv/player_tracker.py
import numpy as np
from typing import List, Dict, Optional
from cv.court_calibration import CourtCalibration


class PlayerTracker:
    """Tracks players using bounding box detections and maintains ID assignments.

    Uses simple IoU-based tracking (ByteTrack integration is Phase 2+ when
    we have real video). For now, tracks by nearest-neighbor matching.
    """

    def __init__(self, calibration: CourtCalibration):
        self.calibration = calibration
        self._tracks: Dict[int, Dict] = {}  # track_id → {x, y, bbox}
        self._player_map: Dict[int, str] = {}  # track_id → player_id
        self._next_track_id = 1
        self._prev_bboxes: Dict[int, np.ndarray] = {}

    def update(self, detections: np.ndarray, frame_number: int) -> List[Dict]:
        """Update with new detections. detections: Nx6 array [x1,y1,x2,y2,conf,cls]."""
        if len(detections) == 0:
            return []

        positions = []
        new_bboxes = {}

        for det in detections:
            x1, y1, x2, y2 = det[0], det[1], det[2], det[3]
            cx = (x1 + x2) / 2.0
            cy_foot = y2  # bottom of bbox = feet position

            court_x, court_y = self.calibration.pixel_to_court(cx, cy_foot)
            bbox = np.array([x1, y1, x2, y2])

            # Match to existing track by nearest bbox
            track_id = self._match_track(bbox)
            if track_id is None:
                track_id = self._next_track_id
                self._next_track_id += 1

            self._tracks[track_id] = {
                "x": float(court_x),
                "y": float(court_y),
                "bbox": bbox,
                "frame": frame_number,
            }
            new_bboxes[track_id] = bbox

            positions.append({
                "track_id": track_id,
                "x": float(court_x),
                "y": float(court_y),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        self._prev_bboxes = new_bboxes
        return positions

    def _match_track(self, bbox: np.ndarray, iou_threshold: float = 0.3) -> Optional[int]:
        best_id = None
        best_iou = iou_threshold

        for tid, prev_bbox in self._prev_bboxes.items():
            iou = self._compute_iou(bbox, prev_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = tid

        return best_id

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def assign_player(self, track_id: int, player_id: str) -> None:
        self._player_map[track_id] = player_id

    def get_player_id(self, track_id: int) -> Optional[str]:
        return self._player_map.get(track_id)

    def get_player_position(self, player_id: str) -> Optional[Dict]:
        for tid, pid in self._player_map.items():
            if pid == player_id and tid in self._tracks:
                return self._tracks[tid]
        return None

    def find_closest_player(self, x: float, y: float) -> Optional[str]:
        min_dist = float("inf")
        closest = None

        for tid, pid in self._player_map.items():
            if tid not in self._tracks:
                continue
            track = self._tracks[tid]
            dist = np.sqrt((track["x"] - x) ** 2 + (track["y"] - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = pid

        return closest
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_player_tracker.py -v`

Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/src/cv/player_tracker.py backend/tests/test_player_tracker.py
git commit -m "feat: player tracker with IoU-based matching, player assignment, and closest-player lookup"
```

---

## Task 6: Backend API (Match Setup + Calibration)

**Files:**
- Create: `backend/tests/test_api.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write API tests**

```python
# backend/tests/test_api.py
import pytest
import json
import os
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create test client with isolated data directory."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import main
    # Override DATA_DIR to use temp directory
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "matches"))
    return TestClient(main.app)


class TestHealthCheck:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


class TestMatchSetup:
    def test_create_match(self, client):
        payload = {
            "match_name": "Test Match",
            "players": {"P1": "Alice", "P2": "Bob", "P3": "Carol", "P4": "Dave"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "match_id" in data

    def test_get_match(self, client):
        payload = {
            "match_name": "Test Match",
            "players": {"P1": "Alice", "P2": "Bob", "P3": "Carol", "P4": "Dave"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        match_id = resp.json()["match_id"]

        resp = client.get(f"/match/{match_id}")
        assert resp.status_code == 200
        assert resp.json()["match_name"] == "Test Match"


class TestCalibration:
    def test_calibrate_court(self, client):
        # Create match first
        payload = {
            "match_name": "Test",
            "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        match_id = resp.json()["match_id"]

        # Calibrate
        corners = {
            "corners": [[320, 200], [1600, 200], [1200, 700], [720, 700]],
        }
        resp = client.post(f"/match/{match_id}/calibrate", json=corners)
        assert resp.status_code == 200
        assert resp.json()["status"] == "calibrated"

    def test_calibrate_rejects_wrong_points(self, client):
        payload = {
            "match_name": "Test",
            "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        match_id = resp.json()["match_id"]

        corners = {"corners": [[0, 0], [1, 1]]}
        resp = client.post(f"/match/{match_id}/calibrate", json=corners)
        assert resp.status_code == 400
```

- [ ] **Step 2: Run test — expect failure**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_api.py -v`

Expected: FAIL — missing endpoints

- [ ] **Step 3: Rewrite main.py**

```python
# backend/main.py
import sys
import os

# Ensure src/ is on Python path for consistent imports (cv.*, models.*, logic.*)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import uuid

app = FastAPI(title="Padel Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/matches"


# --- Request/Response Models ---

class MatchSetupRequest(BaseModel):
    match_name: str
    players: Dict[str, str]
    teams: Dict[str, List[str]]
    golden_point: bool = True
    format: str = "best_of_3"


class CalibrationRequest(BaseModel):
    corners: List[List[float]]


# --- Helpers ---

def _match_dir(match_id: str) -> str:
    return os.path.join(DATA_DIR, match_id)


def _load_match(match_id: str) -> dict:
    path = os.path.join(_match_dir(match_id), "config.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Match not found")
    with open(path) as f:
        return json.load(f)


def _save_match(match_id: str, data: dict):
    d = _match_dir(match_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(data, f, indent=2)


# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "ok", "service": "padel-analyzer"}


@app.post("/match/setup")
def create_match(req: MatchSetupRequest):
    match_id = str(uuid.uuid4())[:8]
    data = {
        "match_id": match_id,
        "match_name": req.match_name,
        "players": req.players,
        "teams": req.teams,
        "golden_point": req.golden_point,
        "format": req.format,
        "calibration": None,
    }
    _save_match(match_id, data)
    return {"match_id": match_id, "status": "created"}


@app.get("/match/{match_id}")
def get_match(match_id: str):
    return _load_match(match_id)


@app.post("/match/{match_id}/calibrate")
def calibrate_court(match_id: str, req: CalibrationRequest):
    if len(req.corners) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 corner points required")

    import numpy as np
    from cv.court_calibration import CourtCalibration

    cal = CourtCalibration()
    cal.calibrate(np.array(req.corners, dtype=np.float32))

    match_data = _load_match(match_id)
    match_data["calibration"] = cal.to_dict()
    _save_match(match_id, match_data)

    return {"status": "calibrated", "match_id": match_id}


# --- Legacy endpoints (kept for backwards compat with current frontend) ---

@app.get("/setup")
def get_setup():
    storage = "court_setup.json"
    if os.path.exists(storage):
        with open(storage) as f:
            return json.load(f)
    return {"name": "Default Court", "cameras": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 4: Run tests — expect pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_api.py -v`

Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_api.py
git commit -m "feat: backend API with match setup, calibration endpoints, and CORS"
```

---

## Task 7: Integration — Wire Pipeline Together

**Files:**
- Modify: `backend/src/cv/video_processor.py`
- Create: `backend/tests/test_integration.py`

- [ ] **Step 1: Write integration tests**

```python
# backend/tests/test_integration.py
import numpy as np
import pytest
from cv.court_calibration import CourtCalibration
from cv.ball_tracker import BallTracker
from cv.player_tracker import PlayerTracker
from logic.scoring_engine import PadelScoringEngine
from models.types import ServerInfo, TeamId


@pytest.fixture
def calibrated_court(sample_court_corners_pixels):
    cal = CourtCalibration()
    cal.calibrate(sample_court_corners_pixels)
    return cal


class TestPipelineIntegration:
    def test_calibration_feeds_ball_tracker(self, calibrated_court):
        tracker = BallTracker(calibrated_court, fps=30)
        pos = tracker.update(bbox=[500, 400, 520, 420], frame_number=0)
        assert pos is not None
        assert calibrated_court.is_in_bounds(pos["x"], pos["y"])

    def test_calibration_feeds_player_tracker(self, calibrated_court):
        tracker = PlayerTracker(calibrated_court)
        dets = np.array([[100, 200, 160, 400, 0.9, 0]])
        positions = tracker.update(dets, frame_number=0)
        assert len(positions) == 1

    def test_full_pipeline_point_flow(self, calibrated_court):
        """Simulate a full point: detection → tracking → scoring."""
        ball_tracker = BallTracker(calibrated_court, fps=30)
        player_tracker = PlayerTracker(calibrated_court)
        engine = PadelScoringEngine(
            first_server=ServerInfo(team_id=TeamId.TEAM_A, player_id="P1"),
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]},
        )

        # Simulate 3 frames of ball movement
        ball_positions = [
            [500, 400, 520, 420],
            [600, 420, 620, 440],
            [700, 450, 720, 470],
        ]
        for i, bbox in enumerate(ball_positions):
            pos = ball_tracker.update(bbox=bbox, frame_number=i)
            assert pos is not None

        # Simulate player detections
        player_dets = np.array([
            [100, 200, 160, 400, 0.9, 0],
            [400, 200, 460, 400, 0.85, 0],
        ])
        player_tracker.update(player_dets, frame_number=0)
        player_tracker.assign_player(track_id=1, player_id="P1")
        player_tracker.assign_player(track_id=2, player_id="P2")

        # Award a point
        engine.add_point(1)
        score = engine.get_score_display()
        assert score["score"] == "15 - 0"

        # Verify trajectory data
        assert len(ball_tracker.trajectory) == 3
        assert ball_tracker.trajectory[-1]["speed"] > 0

    def test_scoring_engine_with_full_config(self):
        engine = PadelScoringEngine(
            golden_point=True,
            sets_to_win=1,
            first_server=ServerInfo(team_id=TeamId.TEAM_A, player_id="P1"),
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]},
        )
        # Play a full set 6-0
        for _ in range(6):
            for _ in range(4):
                engine.add_point(1)
        assert engine.game_over is True
        assert engine.team1_sets == 1
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_integration.py -v`

Expected: All 4 tests PASS

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_integration.py
git commit -m "test: integration tests for calibration → tracking → scoring pipeline"
```

---

## Task 8: Run Full Test Suite & Cleanup

- [ ] **Step 1: Run all tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/ -v --tb=short`

Expected: All tests PASS (approximately 47 tests)

- [ ] **Step 2: Remove deprecated code**

The old `calibration.py` (hardcoded camera params) is replaced by `court_calibration.py`. Add a deprecation notice:

In `backend/src/cv/calibration.py`, add at top:
```python
# DEPRECATED: This module is replaced by court_calibration.py (homography-based).
# Kept for reference by demo_analyzer.py and run_real_analysis.py.
```

- [ ] **Step 3: Run tests one final time**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/ -v --tb=short`

Expected: All tests PASS

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "chore: Phase 1 Foundation complete — deprecation notices, cleanup"
```

---

## Summary

| Task | What it builds | Tests |
|------|---------------|-------|
| 1 | Project structure, shared types, test framework | 10 |
| 2 | Scoring engine: tiebreak, server tracking, match config, let serve | 23 |
| 3 | Homography calibration with zone detection | 11 |
| 4 | Ball tracker with Kalman filter + Z estimation | 9 |
| 5 | Player tracker with IoU matching + assignment | 8 |
| 6 | Backend API (match setup, calibration) | 5 |
| 7 | Integration tests (full pipeline) | 4 |
| 8 | Cleanup + final verification | — |
| **Total** | | **~70 tests** |

**After Phase 1 is complete**, the project has:
- Proper Python package structure with pytest
- Homography-based court calibration (replaces hardcoded camera params)
- Ball detection with Kalman filter smoothing and Z estimation
- Player tracking with ID assignment
- Upgraded scoring engine (tiebreak, server rotation, match format)
- Backend API for match setup and calibration
- Full test coverage

**Phase 2 (next plan)** will add:
- Event detection state machine (IDLE → SERVING → RALLY → POINT_ENDED)
- TrackNetV2 integration for ball detection
- Bounce/serve/fault detection
- WebSocket streaming for live mode
