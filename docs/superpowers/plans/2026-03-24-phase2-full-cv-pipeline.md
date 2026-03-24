# Phase 2: Full CV Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete CV pipeline that turns raw video or live camera into automatically scored padel matches.

**Architecture:** Detection layer (YOLO) → Tracking layer (existing Kalman/IoU) → Event detection (state machine) → Scoring engine (existing) → API + WebSocket → Minimal frontend.

**Tech Stack:** Python 3.14, FastAPI, PyTorch, Ultralytics YOLOv8, OpenCV, React + Three.js

**Spec:** `docs/superpowers/specs/2026-03-24-phase2-full-cv-pipeline-design.md`

**Working directory:** All paths relative to `/Users/jonathan/Documents/Github/padel_analyzer/backend/` unless noted.

**Test command:** `cd /Users/jonathan/Documents/Github/padel_analyzer && source venv/bin/activate && cd backend && python -m pytest tests/ -v --tb=short`

---

## File Map

### New Files
| File | Responsibility |
|------|---------------|
| `src/cv/detectors/__init__.py` | Package init — exports all detectors |
| `src/cv/detectors/base.py` | ABC: `BallDetector`, `PlayerDetector` |
| `src/cv/detectors/device.py` | `get_device()` utility |
| `src/cv/detectors/yolo.py` | `UnifiedYoloDetector`, `YoloBallDetector`, `YoloPlayerDetector` |
| `src/logic/detectors/__init__.py` | Package init |
| `src/logic/detectors/bounce.py` | `BounceDetector` |
| `src/logic/detectors/serve.py` | `ServeDetector` |
| `src/logic/detectors/last_hitter.py` | `LastHitterDetector` |
| `src/logic/detectors/point_end.py` | `PointEndDetector` |
| `src/logic/match_state_machine.py` | `MatchStateMachine` — state transitions |
| `src/logic/event_detector.py` | `EventDetector` — orchestrates sub-detectors + state machine |
| `src/pipeline/__init__.py` | Package init |
| `src/pipeline/video_analyzer.py` | `VideoAnalyzer`, `FrameResult` |
| `src/pipeline/replay_buffer.py` | `ReplayBuffer` — 30s JPEG ring buffer |
| `src/pipeline/live_manager.py` | `LiveManager` — camera feed + WebSocket push |
| `src/models/config.py` | `EventDetectorConfig` dataclass |
| `tests/test_detectors.py` | Tests for YOLO detectors + device util |
| `tests/test_bounce_detector.py` | BounceDetector unit tests |
| `tests/test_serve_detector.py` | ServeDetector unit tests |
| `tests/test_last_hitter.py` | LastHitterDetector unit tests |
| `tests/test_point_end.py` | PointEndDetector unit tests |
| `tests/test_state_machine.py` | MatchStateMachine unit tests |
| `tests/test_event_detector.py` | EventDetector integration tests |
| `tests/test_video_analyzer.py` | VideoAnalyzer pipeline tests |
| `tests/test_replay_buffer.py` | ReplayBuffer unit tests |
| `tests/test_api_phase2.py` | New API endpoint tests |

### Modified Files
| File | Changes |
|------|---------|
| `src/models/types.py` | Add `MatchState` enum, `HIT` event type |
| `main.py` | Add all new API endpoints + WebSocket handler |

---

### Task 1: EventDetectorConfig + Types Updates

**Files:**
- Create: `src/models/config.py`
- Modify: `src/models/types.py:26-34`
- Test: `tests/test_types.py` (add tests)

- [ ] **Step 1: Write failing tests for new types**

In `tests/test_types.py`, add:

```python
def test_match_state_values():
    from models.types import MatchState
    assert MatchState.IDLE.value == "IDLE"
    assert MatchState.SERVING_1ST.value == "SERVING_1ST"
    assert MatchState.SERVING_2ND.value == "SERVING_2ND"
    assert MatchState.RALLY.value == "RALLY"
    assert MatchState.POINT_ENDED.value == "POINT_ENDED"
    assert MatchState.SCORE_UPDATE.value == "SCORE_UPDATE"


def test_event_type_hit():
    from models.types import EventType
    assert EventType.HIT.value == "HIT"


def test_event_detector_config_defaults():
    from models.config import EventDetectorConfig
    cfg = EventDetectorConfig()
    assert cfg.bounce_z_threshold == 0.3
    assert cfg.bounce_speed_dip_pct == 0.4
    assert cfg.serve_timeout_frames == 90
    assert cfg.winner_timeout_frames == 60
    assert cfg.ball_stopped_frames == 15
    assert cfg.auto_assign_after_frames == 30


def test_event_detector_config_custom():
    from models.config import EventDetectorConfig
    cfg = EventDetectorConfig(bounce_z_threshold=0.5, winner_timeout_frames=90)
    assert cfg.bounce_z_threshold == 0.5
    assert cfg.winner_timeout_frames == 90
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_types.py -v --tb=short -k "match_state or event_type_hit or event_detector_config"`
Expected: FAIL — `MatchState` not defined, `config` module not found

- [ ] **Step 3: Add MatchState enum and HIT event type**

In `src/models/types.py`, add after `EventType.POINT_END`:

```python
    HIT = "HIT"
```

Add new enum after `EventType`:

```python
class MatchState(Enum):
    IDLE = "IDLE"
    SERVING_1ST = "SERVING_1ST"
    SERVING_2ND = "SERVING_2ND"
    RALLY = "RALLY"
    POINT_ENDED = "POINT_ENDED"
    SCORE_UPDATE = "SCORE_UPDATE"
```

- [ ] **Step 4: Create EventDetectorConfig**

Create `src/models/config.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_types.py -v --tb=short`
Expected: All pass (previous 9 + 4 new = 13)

- [ ] **Step 6: Commit**

```bash
git add src/models/config.py src/models/types.py tests/test_types.py
git commit -m "feat: add MatchState enum, HIT event type, and EventDetectorConfig"
```

---

### Task 2: Device Auto-Detection Utility

**Files:**
- Create: `src/cv/detectors/__init__.py`, `src/cv/detectors/device.py`
- Test: `tests/test_detectors.py`

- [ ] **Step 1: Write failing test for get_device**

Create `tests/test_detectors.py`:

```python
import pytest


class TestGetDevice:
    def test_returns_string(self):
        from cv.detectors.device import get_device
        device = get_device()
        assert isinstance(device, str)
        assert device in ("mps", "cuda", "cpu")

    def test_cpu_fallback(self, monkeypatch):
        import torch
        from cv.detectors.device import get_device
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device = get_device()
        assert device == "cpu"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detectors.py::TestGetDevice -v --tb=short`
Expected: FAIL — module not found

- [ ] **Step 3: Implement get_device**

Create `src/cv/detectors/__init__.py` (empty file).

Create `src/cv/detectors/device.py`:

```python
import torch


def get_device() -> str:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_detectors.py::TestGetDevice -v --tb=short`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/cv/detectors/ tests/test_detectors.py
git commit -m "feat: device auto-detection utility (MPS > CUDA > CPU)"
```

---

### Task 3: Abstract Detector Interfaces + YOLO Implementation

**Files:**
- Create: `src/cv/detectors/base.py`, `src/cv/detectors/yolo.py`
- Modify: `tests/test_detectors.py`

- [ ] **Step 1: Write failing tests for detectors**

Add to `tests/test_detectors.py`:

```python
from unittest.mock import MagicMock, patch
import numpy as np


class TestUnifiedYoloDetector:
    def test_caches_results_for_same_frame(self):
        from cv.detectors.yolo import UnifiedYoloDetector
        detector = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        mock_model = MagicMock()
        mock_results = [MagicMock()]
        mock_model.return_value = mock_results
        detector.model = mock_model
        detector._cache = (None, None)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = detector.run(frame, frame_id=0)
        r2 = detector.run(frame, frame_id=0)
        assert mock_model.call_count == 1  # only one inference call

    def test_new_frame_runs_inference(self):
        from cv.detectors.yolo import UnifiedYoloDetector
        detector = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        mock_model = MagicMock()
        mock_model.return_value = [MagicMock()]
        detector.model = mock_model
        detector._cache = (None, None)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.run(frame, frame_id=0)
        detector.run(frame, frame_id=1)
        assert mock_model.call_count == 2


class TestYoloBallDetector:
    def _make_mock_result(self, boxes_data):
        """Helper: create mock YOLO result with given boxes."""
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        if len(boxes_data) == 0:
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([])
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([])
        else:
            xyxy = np.array([b[:4] for b in boxes_data])
            cls = np.array([b[4] for b in boxes_data])
            conf = np.array([b[5] for b in boxes_data])
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = conf
        mock_result.boxes = mock_boxes
        return mock_result

    def test_detects_ball_cls32(self):
        from cv.detectors.yolo import YoloBallDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        # Box: [x1, y1, x2, y2, cls, conf]
        result = self._make_mock_result([[100, 200, 120, 220, 32, 0.8]])
        unified.model.return_value = [result]

        detector = YoloBallDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame, frame_id=0)
        assert bbox is not None
        assert len(bbox) == 4

    def test_returns_none_when_no_ball(self):
        from cv.detectors.yolo import YoloBallDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([[100, 200, 300, 400, 0, 0.9]])  # person, not ball
        unified.model.return_value = [result]

        detector = YoloBallDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame, frame_id=0)
        assert bbox is None

    def test_filters_low_confidence(self):
        from cv.detectors.yolo import YoloBallDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([[100, 200, 120, 220, 32, 0.1]])  # low conf
        unified.model.return_value = [result]

        detector = YoloBallDetector(unified, conf_threshold=0.3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame, frame_id=0)
        assert bbox is None


class TestYoloPlayerDetector:
    def _make_mock_result(self, boxes_data):
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        if len(boxes_data) == 0:
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([])
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([])
        else:
            xyxy = np.array([b[:4] for b in boxes_data])
            cls = np.array([b[4] for b in boxes_data])
            conf = np.array([b[5] for b in boxes_data])
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = conf
        mock_result.boxes = mock_boxes
        return mock_result

    def test_detects_persons_cls0(self):
        from cv.detectors.yolo import YoloPlayerDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([
            [100, 200, 200, 400, 0, 0.9],
            [300, 200, 400, 400, 0, 0.85],
        ])
        unified.model.return_value = [result]

        detector = YoloPlayerDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_id=0)
        assert detections.shape[0] == 2
        assert detections.shape[1] == 6

    def test_max_4_detections(self):
        from cv.detectors.yolo import YoloPlayerDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([
            [100, 200, 200, 400, 0, 0.9],
            [300, 200, 400, 400, 0, 0.85],
            [500, 200, 600, 400, 0, 0.8],
            [700, 200, 800, 400, 0, 0.75],
            [900, 200, 1000, 400, 0, 0.6],  # 5th person — should be dropped
        ])
        unified.model.return_value = [result]

        detector = YoloPlayerDetector(unified, max_detections=4)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_id=0)
        assert detections.shape[0] == 4

    def test_returns_empty_when_no_persons(self):
        from cv.detectors.yolo import YoloPlayerDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([[100, 200, 120, 220, 32, 0.8]])  # ball, not person
        unified.model.return_value = [result]

        detector = YoloPlayerDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_id=0)
        assert detections.shape[0] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_detectors.py -v --tb=short`
Expected: FAIL — modules not found

- [ ] **Step 3: Implement abstract base classes**

Create `src/cv/detectors/base.py`:

```python
from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np


class BallDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        """Detect ball in frame. Returns [x1,y1,x2,y2] bbox or None."""
        ...

    @abstractmethod
    def warm_up(self) -> None:
        """Run dummy inference to avoid first-frame latency."""
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        ...


class PlayerDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray, frame_id: int = 0) -> np.ndarray:
        """Detect players in frame. Returns N×6 array [x1,y1,x2,y2,conf,cls]."""
        ...

    @abstractmethod
    def warm_up(self) -> None:
        ...

    @property
    @abstractmethod
    def device(self) -> str:
        ...
```

- [ ] **Step 4: Implement YOLO detectors**

Create `src/cv/detectors/yolo.py`:

```python
import numpy as np
from typing import Optional, List
from ultralytics import YOLO
from cv.detectors.base import BallDetector, PlayerDetector
from cv.detectors.device import get_device


class UnifiedYoloDetector:
    """Runs YOLO once per frame and caches results for both ball and player detectors."""

    def __init__(self, model_path: str = "yolov8n.pt"):
        self._device = get_device()
        self.model = YOLO(model_path)
        self.model.to(self._device)
        self._cache = (None, None)  # (frame_id, results)

    def run(self, frame: np.ndarray, frame_id: int):
        if frame_id != self._cache[0]:
            results = self.model(frame, verbose=False)
            self._cache = (frame_id, results[0])
        return self._cache[1]

    @property
    def device(self) -> str:
        return self._device

    def warm_up(self) -> None:
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)


class YoloBallDetector(BallDetector):
    CLS_SPORTS_BALL = 32

    def __init__(self, unified: UnifiedYoloDetector, conf_threshold: float = 0.3):
        self._unified = unified
        self._conf_threshold = conf_threshold

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> Optional[List[float]]:
        result = self._unified.run(frame, frame_id)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        mask = (cls == self.CLS_SPORTS_BALL) & (conf >= self._conf_threshold)
        if not mask.any():
            return None

        filtered_conf = conf[mask]
        filtered_xyxy = xyxy[mask]
        best_idx = filtered_conf.argmax()
        return filtered_xyxy[best_idx].tolist()

    def warm_up(self) -> None:
        self._unified.warm_up()

    @property
    def device(self) -> str:
        return self._unified.device


class YoloPlayerDetector(PlayerDetector):
    CLS_PERSON = 0

    def __init__(self, unified: UnifiedYoloDetector,
                 conf_threshold: float = 0.5, max_detections: int = 4):
        self._unified = unified
        self._conf_threshold = conf_threshold
        self._max_detections = max_detections

    def detect(self, frame: np.ndarray, frame_id: int = 0) -> np.ndarray:
        result = self._unified.run(frame, frame_id)
        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        conf = boxes.conf.cpu().numpy()

        mask = (cls == self.CLS_PERSON) & (conf >= self._conf_threshold)
        if not mask.any():
            return np.empty((0, 6))

        filtered_xyxy = xyxy[mask]
        filtered_cls = cls[mask]
        filtered_conf = conf[mask]

        # Sort by confidence descending, take top N
        sort_idx = filtered_conf.argsort()[::-1][:self._max_detections]
        out = np.column_stack([
            filtered_xyxy[sort_idx],
            filtered_conf[sort_idx],
            filtered_cls[sort_idx],
        ])
        return out

    def warm_up(self) -> None:
        self._unified.warm_up()

    @property
    def device(self) -> str:
        return self._unified.device
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_detectors.py -v --tb=short`
Expected: All pass

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All 68 + new tests pass

- [ ] **Step 7: Commit**

```bash
git add src/cv/detectors/ tests/test_detectors.py
git commit -m "feat: YOLO detection layer with unified inference and abstract interfaces"
```

---

### Task 4: BounceDetector

**Files:**
- Create: `src/logic/detectors/__init__.py`, `src/logic/detectors/bounce.py`
- Test: `tests/test_bounce_detector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_bounce_detector.py`:

```python
import pytest
from models.config import EventDetectorConfig


@pytest.fixture
def config():
    return EventDetectorConfig()


class TestBounceDetection:
    def test_no_bounce_when_ball_high(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        ball = {"x": 5.0, "y": 5.0, "z": 2.0, "speed": 50.0}
        result = bd.check(ball)
        assert result is None

    def test_bounce_detected_on_z_drop(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        # Simulate ball descending then hitting ground
        for z in [3.0, 2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 5.0, "z": z, "speed": 60.0})
        # Now Z drops below threshold with speed dip
        result = bd.check({"x": 5.0, "y": 5.0, "z": 0.1, "speed": 25.0})
        assert result is not None
        assert result["court_x"] == 5.0
        assert result["court_y"] == 5.0

    def test_bounce_records_court_side(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        for z in [2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 15.0, "z": z, "speed": 60.0})
        result = bd.check({"x": 5.0, "y": 15.0, "z": 0.1, "speed": 25.0})
        assert result is not None
        assert result["side"] == "far"

    def test_bounce_count_increments(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        # First bounce near side
        for z in [2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 5.0, "z": z, "speed": 60.0})
        bd.check({"x": 5.0, "y": 5.0, "z": 0.1, "speed": 25.0})
        assert bd.bounce_count["near"] == 1
        assert bd.bounce_count["far"] == 0

    def test_no_bounce_when_none_ball(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        result = bd.check(None)
        assert result is None

    def test_reset_clears_state(self, config):
        from logic.detectors.bounce import BounceDetector
        bd = BounceDetector(config)
        for z in [2.0, 1.0, 0.5]:
            bd.check({"x": 5.0, "y": 5.0, "z": z, "speed": 60.0})
        bd.check({"x": 5.0, "y": 5.0, "z": 0.1, "speed": 25.0})
        bd.reset()
        assert bd.bounce_count["near"] == 0
        assert bd.bounce_count["far"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bounce_detector.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement BounceDetector**

Create `src/logic/detectors/__init__.py` (empty).

Create `src/logic/detectors/bounce.py`:

```python
from typing import Optional, Dict, List
from models.config import EventDetectorConfig

NET_Y = 10.0  # Net is at y=10m on a 20m court


class BounceDetector:
    def __init__(self, config: EventDetectorConfig):
        self._config = config
        self._z_history: List[float] = []  # sliding window, max 10
        self._speed_history: List[float] = []  # sliding window, max 10
        self.bounce_count: Dict[str, int] = {"near": 0, "far": 0}
        self._last_bounce_frame: int = -10  # debounce

    def check(self, ball_pos: Optional[Dict]) -> Optional[Dict]:
        if ball_pos is None:
            return None

        z = ball_pos.get("z", 0.0)
        speed = ball_pos.get("speed", 0.0)
        self._z_history.append(z)
        self._speed_history.append(speed)
        # Keep sliding window
        if len(self._z_history) > 10:
            self._z_history = self._z_history[-10:]
        if len(self._speed_history) > 10:
            self._speed_history = self._speed_history[-10:]

        if len(self._z_history) < 3:
            return None

        # Ball must be descending (z getting smaller)
        descending = self._z_history[-2] > self._z_history[-1]
        was_higher = self._z_history[-3] > self._config.bounce_z_threshold

        # Z below threshold
        z_low = z <= self._config.bounce_z_threshold

        # Speed dip: current speed < (1 - dip_pct) * recent average
        recent_speeds = self._speed_history[-6:-1] if len(self._speed_history) > 5 else self._speed_history[:-1]
        if not recent_speeds:
            return None
        avg_speed = sum(recent_speeds) / len(recent_speeds)
        speed_dipped = avg_speed > 0 and speed < avg_speed * (1 - self._config.bounce_speed_dip_pct)

        if descending and was_higher and z_low and speed_dipped:
            side = "near" if ball_pos["y"] < NET_Y else "far"
            self.bounce_count[side] += 1
            return {
                "court_x": ball_pos["x"],
                "court_y": ball_pos["y"],
                "side": side,
                "bounce_number": self.bounce_count[side],
            }

        return None

    def reset(self):
        self._z_history.clear()
        self._speed_history.clear()
        self.bounce_count = {"near": 0, "far": 0}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bounce_detector.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/logic/detectors/ tests/test_bounce_detector.py
git commit -m "feat: BounceDetector with Z-drop + speed dip detection"
```

---

### Task 5: LastHitterDetector

**Files:**
- Create: `src/logic/detectors/last_hitter.py`
- Test: `tests/test_last_hitter.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_last_hitter.py`:

```python
import pytest


class TestLastHitterDetector:
    def test_no_hit_on_first_frame(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        ball = {"x": 5.0, "y": 5.0, "z": 1.0, "speed": 50.0}
        players = [{"track_id": 1, "x": 5.0, "y": 5.0, "bbox": []}]
        result = lhd.check(ball, players)
        assert result is None

    def test_detects_hit_on_direction_change(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        players = [
            {"track_id": 1, "x": 3.0, "y": 3.0, "bbox": []},
            {"track_id": 2, "x": 7.0, "y": 15.0, "bbox": []},
        ]
        # Ball moving toward far side
        lhd.check({"x": 5.0, "y": 4.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 8.0, "z": 1.0, "speed": 50.0}, players)
        # Ball reverses direction — hit detected
        result = lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        # Closest player to (5,8) where reversal happened is track_id 2 at (7,15)
        # Actually closest to ball at reversal point
        assert result is not None

    def test_returns_none_when_no_ball(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        result = lhd.check(None, [])
        assert result is None

    def test_last_hitter_stored(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        players = [{"track_id": 1, "x": 5.0, "y": 5.0, "bbox": []}]
        lhd.check({"x": 5.0, "y": 4.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 8.0, "z": 1.0, "speed": 50.0}, players)
        lhd.check({"x": 5.0, "y": 6.0, "z": 1.0, "speed": 50.0}, players)
        assert lhd.last_hitter_track_id is not None

    def test_reset_clears(self):
        from logic.detectors.last_hitter import LastHitterDetector
        lhd = LastHitterDetector()
        lhd.last_hitter_track_id = 1
        lhd.reset()
        assert lhd.last_hitter_track_id is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_last_hitter.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement LastHitterDetector**

Create `src/logic/detectors/last_hitter.py`:

```python
import numpy as np
from typing import Optional, Dict, List


class LastHitterDetector:
    def __init__(self, angle_threshold_deg: float = 90.0):
        self._angle_threshold = np.radians(angle_threshold_deg)
        self._prev_positions: List[Dict] = []
        self.last_hitter_track_id: Optional[int] = None

    def check(self, ball_pos: Optional[Dict], player_positions: List[Dict]) -> Optional[Dict]:
        if ball_pos is None:
            return None

        self._prev_positions.append({"x": ball_pos["x"], "y": ball_pos["y"]})
        if len(self._prev_positions) > 5:
            self._prev_positions = self._prev_positions[-5:]
        if len(self._prev_positions) < 3:
            return None

        # Compute velocity vectors from last 3 positions
        p1 = self._prev_positions[-3]
        p2 = self._prev_positions[-2]
        p3 = self._prev_positions[-1]

        v1 = np.array([p2["x"] - p1["x"], p2["y"] - p1["y"]])
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 0.01 or norm2 < 0.01:
            return None

        cos_angle = np.dot(v1, v2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)

        if angle >= self._angle_threshold:
            # Direction reversal — find closest player to the reversal point
            rev_x, rev_y = p2["x"], p2["y"]
            closest_id = self._find_closest(rev_x, rev_y, player_positions)
            if closest_id is not None:
                self.last_hitter_track_id = closest_id
                return {"track_id": closest_id, "x": rev_x, "y": rev_y}

        return None

    def _find_closest(self, x: float, y: float, players: List[Dict]) -> Optional[int]:
        min_dist = float("inf")
        closest = None
        for p in players:
            dist = np.sqrt((p["x"] - x) ** 2 + (p["y"] - y) ** 2)
            if dist < min_dist:
                min_dist = dist
                closest = p["track_id"]
        return closest

    def reset(self):
        self._prev_positions.clear()
        self.last_hitter_track_id = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_last_hitter.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/logic/detectors/last_hitter.py tests/test_last_hitter.py
git commit -m "feat: LastHitterDetector with velocity direction change detection"
```

---

### Task 6: PointEndDetector

**Files:**
- Create: `src/logic/detectors/point_end.py`
- Test: `tests/test_point_end.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_point_end.py`:

```python
import pytest
from models.config import EventDetectorConfig
from models.types import PointReason


@pytest.fixture
def config():
    return EventDetectorConfig()


class TestDoubleBounce:
    def test_double_bounce_same_side(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        bounce1 = {"side": "near", "bounce_number": 1, "court_x": 3.0, "court_y": 5.0}
        result = ped.check(bounce1, ball_pos={"x": 3.0, "y": 5.0, "z": 0.1, "speed": 10.0}, ball_lost=False)
        assert result is None  # first bounce is fine

        bounce2 = {"side": "near", "bounce_number": 2, "court_x": 4.0, "court_y": 6.0}
        result = ped.check(bounce2, ball_pos={"x": 4.0, "y": 6.0, "z": 0.1, "speed": 10.0}, ball_lost=False)
        assert result is not None
        assert result["reason"] == PointReason.DOUBLE_BOUNCE
        assert result["side"] == "near"


class TestBallOut:
    def test_ball_out_of_enclosure(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        ball = {"x": 12.0, "y": 5.0, "z": 0.1, "speed": 30.0}  # x > 10.5 enclosure
        result = ped.check(None, ball_pos=ball, ball_lost=False)
        assert result is not None
        assert result["reason"] == PointReason.OUT

    def test_ball_inside_enclosure_but_outside_court(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        ball = {"x": 10.3, "y": 5.0, "z": 0.1, "speed": 30.0}  # inside enclosure
        result = ped.check(None, ball_pos=ball, ball_lost=False)
        assert result is None  # legal wall play


class TestBallStopped:
    def test_ball_stopped_after_threshold(self, config):
        from logic.detectors.point_end import PointEndDetector
        config.ball_stopped_frames = 3  # lower for test
        ped = PointEndDetector(config)
        for _ in range(3):
            result = ped.check(None, ball_pos={"x": 5.0, "y": 5.0, "z": 0.0, "speed": 0.1}, ball_lost=False)
        assert result is not None
        assert result["reason"] == PointReason.NET


class TestBallLost:
    def test_ball_lost_triggers_point_end(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        result = ped.check(None, ball_pos=None, ball_lost=True)
        assert result is not None


class TestReset:
    def test_reset_clears_state(self, config):
        from logic.detectors.point_end import PointEndDetector
        ped = PointEndDetector(config)
        ped.check({"side": "near", "bounce_number": 1, "court_x": 3.0, "court_y": 5.0},
                  ball_pos={"x": 3.0, "y": 5.0, "z": 0.1, "speed": 10.0}, ball_lost=False)
        ped.reset()
        assert ped._bounces_per_side == {"near": 0, "far": 0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_point_end.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement PointEndDetector**

Create `src/logic/detectors/point_end.py`:

```python
from typing import Optional, Dict
from models.config import EventDetectorConfig
from models.types import PointReason


NET_Y = 10.0


class PointEndDetector:
    def __init__(self, config: EventDetectorConfig):
        self._config = config
        self._bounces_per_side: Dict[str, int] = {"near": 0, "far": 0}
        self._stopped_frames = 0
        self._frames_since_last_bounce = 0
        self._had_bounce = False
        self._last_bounce_side: Optional[str] = None
        self._bounds = config.enclosure_bounds

    def check(self, bounce: Optional[Dict], ball_pos: Optional[Dict],
              ball_lost: bool) -> Optional[Dict]:

        # 1. Ball lost
        if ball_lost:
            return {"reason": PointReason.OUT, "detail": "ball_lost"}

        if ball_pos is None:
            return None

        x, y = ball_pos.get("x", 0), ball_pos.get("y", 0)

        # 2. Ball out of enclosure
        if (x < self._bounds["x_min"] or x > self._bounds["x_max"] or
                y < self._bounds["y_min"] or y > self._bounds["y_max"]):
            return {"reason": PointReason.OUT, "x": x, "y": y}

        # 3. Wall before bounce: ball near enclosure boundary without
        #    having bounced on this side first
        wall_margin = 0.3
        near_wall = (x <= self._bounds["x_min"] + wall_margin or
                     x >= self._bounds["x_max"] - wall_margin or
                     y <= self._bounds["y_min"] + wall_margin or
                     y >= self._bounds["y_max"] - wall_margin)
        if near_wall and ball_pos.get("z", 0) > 0.5:
            ball_side = "near" if y < NET_Y else "far"
            if self._bounces_per_side[ball_side] == 0 and self._had_bounce:
                return {"reason": PointReason.WALL_BEFORE_BOUNCE, "side": ball_side}

        # 4. Ball stopped
        speed = ball_pos.get("speed", 0)
        if speed < 1.0:
            self._stopped_frames += 1
        else:
            self._stopped_frames = 0

        if self._stopped_frames >= self._config.ball_stopped_frames:
            return {"reason": PointReason.NET, "detail": "ball_stopped"}

        # 5. Double bounce
        if bounce is not None:
            side = bounce["side"]
            self._bounces_per_side[side] += 1
            self._had_bounce = True
            self._last_bounce_side = side
            self._frames_since_last_bounce = 0
            if self._bounces_per_side[side] >= 2:
                return {"reason": PointReason.DOUBLE_BOUNCE, "side": side}
            # Reset opposite side on net crossing
            other = "far" if side == "near" else "near"
            self._bounces_per_side[other] = 0
        else:
            self._frames_since_last_bounce += 1

        # 6. Winner timeout: ball bounced but no return within N frames
        if (self._had_bounce and
                self._frames_since_last_bounce >= self._config.winner_timeout_frames):
            return {"reason": PointReason.WINNER, "side": self._last_bounce_side}

        return None

    def reset(self):
        self._bounces_per_side = {"near": 0, "far": 0}
        self._stopped_frames = 0
        self._frames_since_last_bounce = 0
        self._had_bounce = False
        self._last_bounce_side = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_point_end.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/logic/detectors/point_end.py tests/test_point_end.py
git commit -m "feat: PointEndDetector with double bounce, out, stopped, lost detection"
```

---

### Task 7: ServeDetector

**Files:**
- Create: `src/logic/detectors/serve.py`
- Test: `tests/test_serve_detector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_serve_detector.py`:

```python
import pytest
from unittest.mock import MagicMock
from models.config import EventDetectorConfig
from models.types import ServerInfo, TeamId


@pytest.fixture
def config():
    return EventDetectorConfig()


@pytest.fixture
def mock_calibration():
    cal = MagicMock()
    cal.is_in_service_box.return_value = True
    return cal


@pytest.fixture
def server_near():
    """Server on near side (Team A, P1)."""
    return ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")


class TestServeDetection:
    def test_no_serve_without_server(self, config, mock_calibration):
        from logic.detectors.serve import ServeDetector
        sd = ServeDetector(config, mock_calibration, current_server=None)
        ball = {"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}
        result = sd.check(ball, bounce=None)
        assert result is None

    def test_valid_serve_detected(self, config, mock_calibration, server_near):
        from logic.detectors.serve import ServeDetector
        sd = ServeDetector(config, mock_calibration, current_server=server_near)
        # Ball in service zone near server
        sd.check({"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}, bounce=None)
        sd.check({"x": 5.0, "y": 5.0, "z": 1.5, "speed": 50.0}, bounce=None)
        # Bounce in service box
        bounce = {"court_x": 7.0, "court_y": 13.0, "side": "far", "bounce_number": 1}
        result = sd.check({"x": 7.0, "y": 13.0, "z": 0.1, "speed": 40.0}, bounce=bounce)
        assert result is not None
        assert result["valid"] is True

    def test_fault_wrong_service_box(self, config, server_near):
        from logic.detectors.serve import ServeDetector
        cal = MagicMock()
        cal.is_in_service_box.return_value = False
        sd = ServeDetector(config, cal, current_server=server_near)
        sd.check({"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}, bounce=None)
        sd.check({"x": 5.0, "y": 5.0, "z": 1.5, "speed": 50.0}, bounce=None)
        bounce = {"court_x": 3.0, "court_y": 13.0, "side": "far", "bounce_number": 1}
        result = sd.check({"x": 3.0, "y": 13.0, "z": 0.1, "speed": 40.0}, bounce=bounce)
        assert result is not None
        assert result["valid"] is False
        assert result["fault"] is True

    def test_reset_clears_state(self, config, mock_calibration, server_near):
        from logic.detectors.serve import ServeDetector
        sd = ServeDetector(config, mock_calibration, current_server=server_near)
        sd.check({"x": 5.0, "y": 2.0, "z": 2.0, "speed": 30.0}, bounce=None)
        sd.reset()
        assert sd._serving is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_serve_detector.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement ServeDetector**

Create `src/logic/detectors/serve.py`:

```python
from typing import Optional, Dict
from models.config import EventDetectorConfig
from models.types import ServerInfo, TeamId

NET_Y = 10.0
# Service zones: near-side server serves to far-side service boxes
# Far-side server serves to near-side service boxes
NEAR_SERVICE_ZONE_Y = (0, 4)   # y range where near-side server stands
FAR_SERVICE_ZONE_Y = (16, 20)  # y range where far-side server stands


class ServeDetector:
    def __init__(self, config: EventDetectorConfig, calibration,
                 current_server: Optional[ServerInfo] = None):
        self._config = config
        self._calibration = calibration
        self.current_server = current_server
        self._serving = False
        self._serve_frame_count = 0

    def check(self, ball_pos: Optional[Dict], bounce: Optional[Dict]) -> Optional[Dict]:
        if ball_pos is None or self.current_server is None:
            return None

        x, y = ball_pos["x"], ball_pos["y"]

        # Detect serve start: ball near server position
        if not self._serving:
            if self._is_in_service_zone(y):
                self._serving = True
                self._serve_frame_count = 0
            return None

        self._serve_frame_count += 1

        # Timeout
        if self._serve_frame_count > self._config.serve_timeout_frames:
            self._serving = False
            return {"valid": False, "fault": True, "detail": "serve_timeout"}

        # Check for first bounce during serve
        if bounce is not None:
            target_side = "far" if self._is_near_side_server() else "near"
            if bounce["side"] == target_side:
                # Check if in correct service box
                bx, by = bounce["court_x"], bounce["court_y"]
                in_box = self._calibration.is_in_service_box(bx, by, f"{target_side}_right")
                if not in_box:
                    in_box = self._calibration.is_in_service_box(bx, by, f"{target_side}_left")

                self._serving = False
                if in_box:
                    return {"valid": True, "fault": False}
                else:
                    return {"valid": False, "fault": True, "detail": "wrong_box"}
            else:
                # Bounced on server's own side — fault
                self._serving = False
                return {"valid": False, "fault": True, "detail": "same_side"}

        return None

    def _is_near_side_server(self) -> bool:
        if self.current_server is None:
            return True
        # Team A serves from near side by default
        return self.current_server.team_id == TeamId.TEAM_A

    def _is_in_service_zone(self, y: float) -> bool:
        if self._is_near_side_server():
            return NEAR_SERVICE_ZONE_Y[0] <= y <= NEAR_SERVICE_ZONE_Y[1]
        else:
            return FAR_SERVICE_ZONE_Y[0] <= y <= FAR_SERVICE_ZONE_Y[1]

    def reset(self):
        self._serving = False
        self._serve_frame_count = 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_serve_detector.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/logic/detectors/serve.py tests/test_serve_detector.py
git commit -m "feat: ServeDetector with service box validation and fault detection"
```

---

### Task 8: MatchStateMachine

**Files:**
- Create: `src/logic/match_state_machine.py`
- Test: `tests/test_state_machine.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_state_machine.py`:

```python
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
        sm.on_serve_result(valid=False)  # -> SERVING_2ND
        sm.on_serve_result(valid=False)  # -> POINT_ENDED (double fault)
        assert sm.state == MatchState.POINT_ENDED
        assert sm.point_end_reason == PointReason.DOUBLE_FAULT

    def test_rally_point_end(self):
        from logic.match_state_machine import MatchStateMachine
        sm = MatchStateMachine()
        sm.on_serve_started()
        sm.on_serve_result(valid=True)  # -> RALLY
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
        sm.on_serve_result(valid=False)  # -> SERVING_2ND
        sm.on_let()
        assert sm.state == MatchState.SERVING_2ND
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_state_machine.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement MatchStateMachine**

Create `src/logic/match_state_machine.py`:

```python
from typing import Optional
from models.types import MatchState, PointReason


class MatchStateMachine:
    def __init__(self):
        self.state = MatchState.IDLE
        self.point_end_reason: Optional[PointReason] = None

    def on_serve_started(self):
        if self.state == MatchState.IDLE:
            self.state = MatchState.SERVING_1ST

    def on_serve_result(self, valid: bool):
        if self.state == MatchState.SERVING_1ST:
            if valid:
                self.state = MatchState.RALLY
            else:
                self.state = MatchState.SERVING_2ND
        elif self.state == MatchState.SERVING_2ND:
            if valid:
                self.state = MatchState.RALLY
            else:
                self.point_end_reason = PointReason.DOUBLE_FAULT
                self.state = MatchState.POINT_ENDED

    def on_let(self):
        # Let serve — stay in current serving state
        pass

    def on_point_ended(self, reason: PointReason):
        if self.state == MatchState.RALLY:
            self.point_end_reason = reason
            self.state = MatchState.POINT_ENDED

    def on_score_updated(self):
        if self.state == MatchState.POINT_ENDED:
            self.state = MatchState.IDLE
            self.point_end_reason = None

    def reset_to_idle(self):
        self.state = MatchState.IDLE
        self.point_end_reason = None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_state_machine.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/logic/match_state_machine.py tests/test_state_machine.py
git commit -m "feat: MatchStateMachine with full point lifecycle transitions"
```

---

### Task 9: EventDetector (Orchestrator)

**Files:**
- Create: `src/logic/event_detector.py`
- Test: `tests/test_event_detector.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_event_detector.py`:

```python
import pytest
from unittest.mock import MagicMock
from models.config import EventDetectorConfig
from models.types import MatchState, PointReason, TeamId, ServerInfo


@pytest.fixture
def config():
    return EventDetectorConfig()


@pytest.fixture
def mock_calibration():
    cal = MagicMock()
    cal.is_in_service_box.return_value = True
    cal.is_in_bounds.return_value = True
    return cal


@pytest.fixture
def mock_scoring():
    eng = MagicMock()
    eng.current_server = ServerInfo(team_id=TeamId.TEAM_A, player_id="P1")
    eng.get_score_display.return_value = {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"}
    return eng


@pytest.fixture
def mock_player_tracker():
    pt = MagicMock()
    pt.find_closest_player.return_value = "P1"
    pt.get_player_id.return_value = "P1"
    return pt


@pytest.fixture
def team_map():
    return {"P1": 1, "P2": 1, "P3": 2, "P4": 2}


class TestEventDetector:
    def test_init_state_idle(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        assert ed.state_machine.state == MatchState.IDLE

    def test_process_returns_list(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        ball = {"x": 5.0, "y": 5.0, "z": 2.0, "speed": 50.0}
        players = [{"track_id": 1, "x": 5.0, "y": 5.0, "bbox": []}]
        events = ed.process(ball, players, frame_no=0)
        assert isinstance(events, list)

    def test_process_handles_none_ball(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        events = ed.process(None, [], frame_no=0)
        assert isinstance(events, list)

    def test_point_end_calls_add_point(self, config, mock_calibration, mock_scoring, mock_player_tracker, team_map):
        from logic.event_detector import EventDetector
        ed = EventDetector(config, mock_calibration, mock_scoring, mock_player_tracker, team_map)
        # Force state to RALLY and trigger point end
        ed.state_machine.state = MatchState.RALLY
        ed.state_machine.on_point_ended(PointReason.DOUBLE_BOUNCE)
        # Call resolve to process point end
        ed._resolve_point_end(PointReason.DOUBLE_BOUNCE, "near", last_hitter_track_id=1)
        mock_scoring.add_point.assert_called_once()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_event_detector.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement EventDetector**

Create `src/logic/event_detector.py`:

```python
from typing import Optional, Dict, List
from models.config import EventDetectorConfig
from models.types import MatchState, MatchEvent, EventType, CourtPoint, PointReason, TeamId
from logic.match_state_machine import MatchStateMachine
from logic.detectors.bounce import BounceDetector
from logic.detectors.last_hitter import LastHitterDetector
from logic.detectors.serve import ServeDetector
from logic.detectors.point_end import PointEndDetector


class EventDetector:
    def __init__(self, config: EventDetectorConfig, calibration,
                 scoring_engine, player_tracker, team_map: Dict[str, int]):
        self._config = config
        self._calibration = calibration
        self._scoring_engine = scoring_engine
        self._player_tracker = player_tracker
        self._team_map = team_map

        self.state_machine = MatchStateMachine()
        self._bounce_detector = BounceDetector(config)
        self._last_hitter = LastHitterDetector()
        self._serve_detector = ServeDetector(
            config, calibration,
            current_server=getattr(scoring_engine, 'current_server', None)
        )
        self._point_end_detector = PointEndDetector(config)

    def process(self, ball_pos: Optional[Dict], player_positions: List[Dict],
                frame_no: int) -> List[MatchEvent]:
        events: List[MatchEvent] = []

        # Update serve detector with current server
        self._serve_detector.current_server = getattr(
            self._scoring_engine, 'current_server', None
        )

        # Run sub-detectors
        bounce = self._bounce_detector.check(ball_pos)
        hit = self._last_hitter.check(ball_pos, player_positions)
        ball_lost = ball_pos is None and hasattr(self, '_had_ball') and self._had_ball

        if ball_pos is not None:
            self._had_ball = True

        # State-dependent processing
        state = self.state_machine.state

        if state == MatchState.IDLE:
            # Check for serve start
            if ball_pos is not None and self._serve_detector.current_server is not None:
                serve_check = self._serve_detector.check(ball_pos, bounce)
                if self._serve_detector._serving:
                    self.state_machine.on_serve_started()

        elif state in (MatchState.SERVING_1ST, MatchState.SERVING_2ND):
            if ball_pos is not None:
                serve_result = self._serve_detector.check(ball_pos, bounce)
                if serve_result is not None:
                    self.state_machine.on_serve_result(serve_result["valid"])
                    if serve_result.get("fault"):
                        events.append(self._make_event(
                            EventType.FAULT, frame_no, ball_pos))

                    # Check for double fault point end
                    if self.state_machine.state == MatchState.POINT_ENDED:
                        self._resolve_point_end(
                            PointReason.DOUBLE_FAULT, None,
                            last_hitter_track_id=None)
                        events.append(self._make_event(
                            EventType.POINT_END, frame_no, ball_pos,
                            {"reason": "double_fault"}))
                        self.state_machine.on_score_updated()
                        self._reset_detectors()

                    elif serve_result["valid"]:
                        events.append(self._make_event(
                            EventType.SERVE, frame_no, ball_pos))

        elif state == MatchState.RALLY:
            # Check bounce
            if bounce is not None:
                events.append(self._make_event(
                    EventType.BOUNCE, frame_no, ball_pos,
                    {"side": bounce["side"]}))

            # Check hit
            if hit is not None:
                events.append(self._make_event(
                    EventType.HIT, frame_no, ball_pos,
                    {"track_id": hit["track_id"]}))

            # Check point end
            point_end = self._point_end_detector.check(bounce, ball_pos, ball_lost)
            if point_end is not None:
                reason = point_end["reason"]
                side = point_end.get("side")
                self.state_machine.on_point_ended(reason)
                self._resolve_point_end(
                    reason, side,
                    self._last_hitter.last_hitter_track_id)
                events.append(self._make_event(
                    EventType.POINT_END, frame_no, ball_pos,
                    {"reason": reason.value}))
                self.state_machine.on_score_updated()
                self._reset_detectors()

        return events

    def _resolve_point_end(self, reason: PointReason, side: Optional[str],
                           last_hitter_track_id: Optional[int]):
        """Determine winner and call scoring engine."""
        winner_team = self._determine_winner(reason, side, last_hitter_track_id)
        if winner_team is not None:
            self._scoring_engine.add_point(winner_team, reason)

    def _determine_winner(self, reason: PointReason, side: Optional[str],
                          last_hitter_track_id: Optional[int]) -> Optional[int]:
        if reason == PointReason.DOUBLE_FAULT:
            # Receiving team wins
            server = getattr(self._scoring_engine, 'current_server', None)
            if server:
                return 2 if server.team_id == TeamId.TEAM_A else 1
            return 2

        if reason == PointReason.DOUBLE_BOUNCE:
            # Team on OTHER side wins
            if side == "near":
                return 2  # Team B wins (they're on far side)
            return 1

        if reason in (PointReason.OUT, PointReason.NET, PointReason.WALL_BEFORE_BOUNCE):
            # Team that did NOT hit last wins
            if last_hitter_track_id is not None:
                player_id = self._player_tracker.get_player_id(last_hitter_track_id)
                if player_id and player_id in self._team_map:
                    hitter_team = self._team_map[player_id]
                    return 2 if hitter_team == 1 else 1
            return None

        if reason == PointReason.WINNER:
            # Team that hit last wins
            if last_hitter_track_id is not None:
                player_id = self._player_tracker.get_player_id(last_hitter_track_id)
                if player_id and player_id in self._team_map:
                    return self._team_map[player_id]
            return None

        return None

    def _reset_detectors(self):
        self._bounce_detector.reset()
        self._last_hitter.reset()
        self._serve_detector.reset()
        self._point_end_detector.reset()

    @staticmethod
    def _make_event(event_type: EventType, frame_no: int,
                    ball_pos: Optional[Dict], metadata: Dict = None) -> MatchEvent:
        x = ball_pos["x"] if ball_pos else 0.0
        y = ball_pos["y"] if ball_pos else 0.0
        return MatchEvent(
            event_type=event_type,
            timestamp=ball_pos.get("timestamp", 0.0) if ball_pos else 0.0,
            frame_number=frame_no,
            position=CourtPoint(x=x, y=y),
            metadata=metadata or {},
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_event_detector.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/logic/event_detector.py tests/test_event_detector.py
git commit -m "feat: EventDetector orchestrating bounce/serve/hit/point-end + scoring"
```

---

### Task 10: ReplayBuffer

**Files:**
- Create: `src/pipeline/__init__.py`, `src/pipeline/replay_buffer.py`
- Test: `tests/test_replay_buffer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_replay_buffer.py`:

```python
import pytest
import numpy as np


class TestReplayBuffer:
    def test_add_frame(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=10)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rb.add(frame, timestamp=0.0)
        assert len(rb) == 1

    def test_ring_buffer_wraps(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=3)
        for i in range(5):
            frame = np.ones((480, 640, 3), dtype=np.uint8) * i
            rb.add(frame, timestamp=float(i))
        assert len(rb) == 3

    def test_get_frames_ordered(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=5)
        for i in range(5):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * i
            rb.add(frame, timestamp=float(i))
        frames = rb.get_frames()
        assert len(frames) == 5
        assert frames[0]["timestamp"] == 0.0
        assert frames[4]["timestamp"] == 4.0

    def test_get_frames_after_wrap(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=3)
        for i in range(5):
            frame = np.ones((100, 100, 3), dtype=np.uint8) * i
            rb.add(frame, timestamp=float(i))
        frames = rb.get_frames()
        assert len(frames) == 3
        # Should contain timestamps 2, 3, 4 (oldest first)
        assert frames[0]["timestamp"] == 2.0
        assert frames[2]["timestamp"] == 4.0

    def test_frames_stored_as_jpeg(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=5, jpeg_quality=70)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rb.add(frame, timestamp=0.0)
        frames = rb.get_frames()
        assert isinstance(frames[0]["jpeg"], bytes)

    def test_clear(self):
        from pipeline.replay_buffer import ReplayBuffer
        rb = ReplayBuffer(max_frames=5)
        rb.add(np.zeros((100, 100, 3), dtype=np.uint8), 0.0)
        rb.clear()
        assert len(rb) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_replay_buffer.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement ReplayBuffer**

Create `src/pipeline/__init__.py` (empty).

Create `src/pipeline/replay_buffer.py`:

```python
import cv2
import numpy as np
from typing import List, Dict


class ReplayBuffer:
    def __init__(self, max_frames: int = 900, jpeg_quality: int = 70):
        self._max = max_frames
        self._quality = jpeg_quality
        self._buffer: List[Dict] = []
        self._index = 0
        self._full = False

    def add(self, frame: np.ndarray, timestamp: float):
        _, jpeg = cv2.imencode('.jpg', frame,
                               [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        entry = {"jpeg": jpeg.tobytes(), "timestamp": timestamp}

        if not self._full:
            self._buffer.append(entry)
            if len(self._buffer) >= self._max:
                self._full = True
                self._index = 0
        else:
            self._buffer[self._index] = entry
            self._index = (self._index + 1) % self._max

    def get_frames(self) -> List[Dict]:
        if not self._full:
            return list(self._buffer)
        # Return in chronological order: from index to end, then start to index
        return self._buffer[self._index:] + self._buffer[:self._index]

    def __len__(self):
        return len(self._buffer)

    def clear(self):
        self._buffer.clear()
        self._index = 0
        self._full = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_replay_buffer.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/pipeline/ tests/test_replay_buffer.py
git commit -m "feat: ReplayBuffer with 30s JPEG ring buffer"
```

---

### Task 11: VideoAnalyzer Pipeline

**Files:**
- Create: `src/pipeline/video_analyzer.py`
- Test: `tests/test_video_analyzer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_video_analyzer.py`:

```python
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from models.config import EventDetectorConfig


@pytest.fixture
def mock_deps():
    """Create mocked dependencies for VideoAnalyzer."""
    calibration = MagicMock()
    calibration.pixel_to_court.return_value = (5.0, 10.0)
    calibration.is_in_bounds.return_value = True
    calibration.is_in_service_box.return_value = True

    config = EventDetectorConfig()
    return calibration, config


class TestVideoAnalyzer:
    def test_process_frame_returns_frame_result(self, mock_deps):
        from pipeline.video_analyzer import VideoAnalyzer, FrameResult
        calibration, config = mock_deps

        with patch('pipeline.video_analyzer.UnifiedYoloDetector') as MockUnified:
            mock_unified = MockUnified.return_value
            mock_unified.run.return_value = MagicMock(boxes=MagicMock(
                xyxy=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([]).reshape(0, 4))))),
                cls=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([]))))),
                conf=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([]))))),
            ))

            va = VideoAnalyzer(
                match_id="test",
                calibration=calibration,
                config=config,
            )
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = va.process_frame(frame, frame_no=0)
            assert isinstance(result, FrameResult)
            assert result.frame_number == 0

    def test_auto_assignment_after_n_frames(self, mock_deps):
        from pipeline.video_analyzer import VideoAnalyzer
        calibration, config = mock_deps
        config.auto_assign_after_frames = 2

        with patch('pipeline.video_analyzer.UnifiedYoloDetector') as MockUnified:
            mock_unified = MockUnified.return_value
            # Return 4 players
            mock_result = MagicMock()
            xyxy = np.array([[100,200,200,400],[300,200,400,400],[500,200,600,400],[700,200,800,400]])
            cls = np.array([0,0,0,0])
            conf = np.array([0.9,0.85,0.8,0.75])
            mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_result.boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_result.boxes.conf.cpu.return_value.numpy.return_value = conf
            mock_unified.run.return_value = mock_result

            va = VideoAnalyzer(match_id="test", calibration=calibration, config=config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            va.process_frame(frame, 0)
            assert not va._auto_assigned
            va.process_frame(frame, 1)
            assert not va._auto_assigned
            va.process_frame(frame, 2)
            assert va._auto_assigned
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_video_analyzer.py -v --tb=short`
Expected: FAIL

- [ ] **Step 3: Implement VideoAnalyzer**

Create `src/pipeline/video_analyzer.py`:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable
import numpy as np
import cv2

from models.config import EventDetectorConfig
from models.types import MatchEvent, MatchConfig, ServerInfo, TeamId
from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector, YoloPlayerDetector
from cv.ball_tracker import BallTracker
from cv.player_tracker import PlayerTracker
from logic.event_detector import EventDetector
from logic.scoring_engine import PadelScoringEngine


@dataclass
class FrameResult:
    ball_position: Optional[Dict]
    player_positions: List[Dict]
    events: List[MatchEvent]
    score: Dict
    frame_number: int


class VideoAnalyzer:
    def __init__(self, match_id: str, calibration,
                 config: EventDetectorConfig = None,
                 match_config: MatchConfig = None):
        config = config or EventDetectorConfig()
        match_config = match_config or MatchConfig()

        # Detection
        unified = UnifiedYoloDetector()
        self.ball_detector = YoloBallDetector(unified)
        self.player_detector = YoloPlayerDetector(unified)

        # Tracking
        self.ball_tracker = BallTracker(calibration)
        self.player_tracker = PlayerTracker(calibration)

        # Scoring
        team_players = match_config.teams if match_config.teams else None
        first_server = match_config.first_server if match_config.first_server else None
        self.scoring_engine = PadelScoringEngine(
            golden_point=match_config.golden_point,
            sets_to_win=match_config.format.value if match_config.format else 2,
            first_server=first_server,
            team_players=team_players,
        )

        # Team map
        team_map = {}
        if match_config.teams:
            for team_id, players in match_config.teams.items():
                tid = team_id.value if isinstance(team_id, TeamId) else team_id
                for pid in players:
                    team_map[pid] = tid
        if not team_map:
            team_map = {"P1": 1, "P2": 1, "P3": 2, "P4": 2}

        # Event detection
        self.event_detector = EventDetector(
            config, calibration, self.scoring_engine,
            self.player_tracker, team_map
        )

        self._config = config
        self._match_id = match_id
        self._auto_assigned = False
        self._frame_count = 0
        self.all_events: List[MatchEvent] = []

    def process_frame(self, frame: np.ndarray, frame_no: int) -> FrameResult:
        self._frame_count = frame_no
        ball_bbox = self.ball_detector.detect(frame, frame_no)
        player_detections = self.player_detector.detect(frame, frame_no)
        ball_pos = self.ball_tracker.update(ball_bbox, frame_no)
        player_pos = self.player_tracker.update(player_detections, frame_no)

        # Auto-assign players
        if not self._auto_assigned and frame_no >= self._config.auto_assign_after_frames:
            self._auto_assign_players(player_pos)
            self._auto_assigned = True

        events = self.event_detector.process(ball_pos, player_pos, frame_no)
        self.all_events.extend(events)

        return FrameResult(
            ball_position=ball_pos,
            player_positions=player_pos,
            events=events,
            score=self.scoring_engine.get_score_display(),
            frame_number=frame_no,
        )

    def analyze_video(self, video_path: str,
                      progress_callback: Optional[Callable] = None) -> Dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ball_tracker.fps = fps
        self.ball_tracker.dt = 1.0 / fps

        frame_no = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            self.process_frame(frame, frame_no)
            frame_no += 1

            if progress_callback and frame_no % 30 == 0:
                pct = (frame_no / total_frames * 100) if total_frames > 0 else 0
                progress_callback(frame_no, total_frames, pct)

        cap.release()
        return {
            "match_id": self._match_id,
            "frames_processed": frame_no,
            "events": len(self.all_events),
            "final_score": self.scoring_engine.get_score_display(),
        }

    def _auto_assign_players(self, current_positions: List[Dict]):
        if len(current_positions) < 2:
            return

        near = sorted([p for p in current_positions if p["y"] < 10.0],
                      key=lambda p: p["x"])
        far = sorted([p for p in current_positions if p["y"] >= 10.0],
                     key=lambda p: p["x"])

        assignments = []
        for i, p in enumerate(near[:2]):
            pid = f"P{i + 1}"
            assignments.append((p["track_id"], pid))
        for i, p in enumerate(far[:2]):
            pid = f"P{i + 3}"
            assignments.append((p["track_id"], pid))

        for track_id, player_id in assignments:
            self.player_tracker.assign_player(track_id, player_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_video_analyzer.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/pipeline/video_analyzer.py tests/test_video_analyzer.py
git commit -m "feat: VideoAnalyzer pipeline with offline analysis and auto-assignment"
```

---

### Task 12: API Endpoints — Offline Analysis

**Files:**
- Modify: `main.py`
- Test: `tests/test_api_phase2.py`

- [ ] **Step 1: Write failing tests for new endpoints**

Create `tests/test_api_phase2.py`:

```python
import pytest
from httpx import AsyncClient, ASGITransport
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from main import app


@pytest.fixture
def transport():
    return ASGITransport(app=app)


class TestScoreEndpoint:
    @pytest.mark.asyncio
    async def test_get_score_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/match/nonexistent/score")
            assert resp.status_code == 404


class TestEventsEndpoint:
    @pytest.mark.asyncio
    async def test_get_events_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/match/nonexistent/events")
            assert resp.status_code == 404


class TestTrajectoryEndpoint:
    @pytest.mark.asyncio
    async def test_get_trajectory_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/match/nonexistent/trajectory")
            assert resp.status_code == 404


class TestCorrectScore:
    @pytest.mark.asyncio
    async def test_correct_score_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/match/nonexistent/correct-score",
                                     json={"team": 1})
            assert resp.status_code == 404


class TestAssignPlayer:
    @pytest.mark.asyncio
    async def test_assign_player_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/match/nonexistent/assign-player",
                                     json={"track_id": 1, "player_id": "P1"})
            assert resp.status_code == 404
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_api_phase2.py -v --tb=short`
Expected: FAIL — endpoints not found (405 or 404)

- [ ] **Step 3: Add new endpoints to main.py**

Add to `main.py` after the calibrate endpoint:

```python
# ── In-memory state for active analyses ─────────────────────
_active_analyzers: Dict[str, object] = {}


class CorrectScoreRequest(BaseModel):
    team: int


class AssignPlayerRequest(BaseModel):
    track_id: int
    player_id: str


@app.get("/match/{match_id}/score")
def get_score(match_id: str):
    _load_match(match_id)  # verify exists
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"}
    return analyzer.scoring_engine.get_score_display()


@app.get("/match/{match_id}/events")
def get_events(match_id: str):
    _load_match(match_id)  # verify exists
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"events": []}
    return {"events": [
        {
            "event_type": e.event_type.value,
            "timestamp": e.timestamp,
            "frame_number": e.frame_number,
            "position": {"x": e.position.x, "y": e.position.y},
            "metadata": e.metadata,
        }
        for e in analyzer.all_events
    ]}


@app.get("/match/{match_id}/trajectory")
def get_trajectory(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"trajectory": []}
    return {"trajectory": analyzer.ball_tracker.trajectory}


@app.get("/match/{match_id}/stats")
def get_stats(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"stats": {}}
    return {"stats": {
        "total_events": len(analyzer.all_events),
        "frames_processed": analyzer._frame_count if hasattr(analyzer, '_frame_count') else 0,
    }}


@app.post("/match/{match_id}/correct-score")
def correct_score(match_id: str, req: CorrectScoreRequest):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        raise HTTPException(status_code=400, detail="No active analysis for this match")
    analyzer.scoring_engine.add_point(req.team)
    return {"status": "corrected", "score": analyzer.scoring_engine.get_score_display()}


@app.post("/match/{match_id}/assign-player")
def assign_player(match_id: str, req: AssignPlayerRequest):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        raise HTTPException(status_code=400, detail="No active analysis for this match")
    analyzer.player_tracker.assign_player(req.track_id, req.player_id)
    return {"status": "assigned", "track_id": req.track_id, "player_id": req.player_id}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_api_phase2.py -v --tb=short`
Expected: All pass

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_api_phase2.py
git commit -m "feat: API endpoints for score, events, trajectory, stats, corrections"
```

---

### Task 13: Offline Analysis Endpoints (Upload + Start + Status)

**Files:**
- Modify: `main.py`
- Modify: `tests/test_api_phase2.py`

- [ ] **Step 1: Write failing tests for upload/analyze endpoints**

Add to `tests/test_api_phase2.py`:

```python
class TestAnalyzeUpload:
    @pytest.mark.asyncio
    async def test_analyze_status_no_job(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analyze/status/nonexistent")
            assert resp.status_code == 404
```

- [ ] **Step 2: Add upload, start, and status endpoints to main.py**

Add to `main.py`:

```python
from fastapi import UploadFile, File, BackgroundTasks
import shutil

_analysis_jobs: Dict[str, Dict] = {}  # job_id -> {state, percent, match_id, ...}


@app.post("/analyze/upload")
async def upload_video(match_id: str, file: UploadFile = File(...)):
    _load_match(match_id)  # verify match exists
    match_dir = _match_dir(match_id)
    os.makedirs(match_dir, exist_ok=True)
    video_path = os.path.join(match_dir, "video.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = match_id  # use match_id as job_id for simplicity
    _analysis_jobs[job_id] = {"state": "uploaded", "percent": 0, "match_id": match_id}
    return {"job_id": job_id, "status": "uploaded"}


@app.post("/analyze/start/{job_id}")
def start_analysis(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _analysis_jobs[job_id]
    match_id = job["match_id"]
    match_data = _load_match(match_id)

    import numpy as np
    from cv.court_calibration import CourtCalibration
    from models.config import EventDetectorConfig
    from pipeline.video_analyzer import VideoAnalyzer

    cal = CourtCalibration()
    if match_data.get("calibration"):
        cal = CourtCalibration.from_dict(match_data["calibration"])

    config = EventDetectorConfig()
    analyzer = VideoAnalyzer(match_id=match_id, calibration=cal, config=config)
    _active_analyzers[match_id] = analyzer

    video_path = os.path.join(_match_dir(match_id), "video.mp4")

    def progress_cb(frame, total, pct):
        _analysis_jobs[job_id]["percent"] = round(pct, 1)
        _analysis_jobs[job_id]["state"] = "processing"

    def run_analysis():
        _analysis_jobs[job_id]["state"] = "processing"
        try:
            result = analyzer.analyze_video(video_path, progress_callback=progress_cb)
            _analysis_jobs[job_id]["state"] = "complete"
            _analysis_jobs[job_id]["percent"] = 100
        except Exception as e:
            _analysis_jobs[job_id]["state"] = "error"
            _analysis_jobs[job_id]["error"] = str(e)

    background_tasks.add_task(run_analysis)
    return {"status": "started", "job_id": job_id}


@app.get("/analyze/status/{job_id}")
def get_analysis_status(job_id: str):
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _analysis_jobs[job_id]
```

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_api_phase2.py -v --tb=short`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add main.py tests/test_api_phase2.py
git commit -m "feat: offline analysis endpoints — upload, start, status polling"
```

---

### Task 14: LiveManager + WebSocket (was Task 13)

**Files:**
- Create: `src/pipeline/live_manager.py`
- Modify: `main.py` — add WebSocket handler and live endpoints

- [ ] **Step 1: Implement LiveManager**

Create `src/pipeline/live_manager.py`:

```python
import asyncio
import base64
import cv2
import numpy as np
import threading
from typing import Optional, Dict, List, Set
from pipeline.video_analyzer import VideoAnalyzer, FrameResult
from pipeline.replay_buffer import ReplayBuffer


class LiveManager:
    def __init__(self, analyzer: VideoAnalyzer, device_id=0,
                 record: bool = False, record_path: str = None):
        self._analyzer = analyzer
        self._device_id = device_id
        self._replay_buffer = ReplayBuffer(max_frames=900)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_no = 0
        self._latest_result: Optional[FrameResult] = None
        self._latest_jpeg: Optional[bytes] = None
        self._record = record
        self._record_path = record_path
        self._writer: Optional[cv2.VideoWriter] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self.fps: float = 0.0

    def start(self):
        self._cap = cv2.VideoCapture(self._device_id)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera: {self._device_id}")
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()

    def _run_loop(self):
        import time
        retry_count = 0
        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                retry_count += 1
                if retry_count >= 3:
                    self._running = False
                    break
                time.sleep(1)
                continue
            retry_count = 0

            t0 = time.monotonic()
            result = self._analyzer.process_frame(frame, self._frame_no)
            self._latest_result = result
            self._frame_no += 1

            # JPEG encode for streaming
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self._latest_jpeg = jpeg.tobytes()

            # Replay buffer
            self._replay_buffer.add(frame, timestamp=self._frame_no / 30.0)

            # Recording
            if self._record and self._writer is None and self._record_path:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self._writer = cv2.VideoWriter(self._record_path, fourcc, 30.0, (w, h))
            if self._writer:
                self._writer.write(frame)

            # Push events to queue
            if result.events:
                for event in result.events:
                    try:
                        self._event_queue.put_nowait({
                            "type": "event",
                            "data": {
                                "event_type": event.event_type.value,
                                "timestamp": event.timestamp,
                                "frame": event.frame_number,
                            }
                        })
                    except asyncio.QueueFull:
                        pass

            elapsed = time.monotonic() - t0
            self.fps = 1.0 / elapsed if elapsed > 0 else 0.0

    def get_latest_frame_b64(self) -> Optional[str]:
        if self._latest_jpeg:
            return base64.b64encode(self._latest_jpeg).decode()
        return None

    def get_replay(self) -> List[Dict]:
        return self._replay_buffer.get_frames()

    @property
    def is_running(self) -> bool:
        return self._running
```

- [ ] **Step 2: Add live endpoints and WebSocket to main.py**

Add to `main.py`:

```python
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
import asyncio


class LiveStartRequest(BaseModel):
    device_id: int = 0
    rtsp_url: Optional[str] = None
    match_id: str
    record: bool = False


_live_manager = None


@app.post("/live/start")
def start_live(req: LiveStartRequest):
    global _live_manager
    match_data = _load_match(req.match_id)

    import numpy as np
    from cv.court_calibration import CourtCalibration
    from models.config import EventDetectorConfig
    from pipeline.video_analyzer import VideoAnalyzer
    from pipeline.live_manager import LiveManager

    cal = CourtCalibration()
    if match_data.get("calibration"):
        cal = CourtCalibration.from_dict(match_data["calibration"])

    config = EventDetectorConfig()
    analyzer = VideoAnalyzer(match_id=req.match_id, calibration=cal, config=config)
    _active_analyzers[req.match_id] = analyzer

    device = req.rtsp_url if req.rtsp_url else req.device_id
    record_path = os.path.join(_match_dir(req.match_id), "recording.mp4") if req.record else None

    _live_manager = LiveManager(analyzer, device_id=device,
                                record=req.record, record_path=record_path)
    _live_manager.start()
    return {"status": "started", "match_id": req.match_id}


@app.post("/live/stop")
def stop_live():
    global _live_manager
    if _live_manager is None:
        raise HTTPException(status_code=400, detail="No live session running")
    _live_manager.stop()
    _live_manager = None
    return {"status": "stopped"}


@app.get("/live/replay")
def get_replay():
    if _live_manager is None:
        raise HTTPException(status_code=400, detail="No live session running")
    import base64
    frames = _live_manager.get_replay()
    return {"frames": [
        {"jpeg": base64.b64encode(f["jpeg"]).decode(), "timestamp": f["timestamp"]}
        for f in frames
    ]}


@app.websocket("/live/stream")
async def live_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            if _live_manager and _live_manager.is_running:
                frame_b64 = _live_manager.get_latest_frame_b64()
                if frame_b64:
                    await ws.send_json({"type": "frame", "jpeg": frame_b64})

                # Check for events
                try:
                    event = _live_manager._event_queue.get_nowait()
                    await ws.send_json(event)
                except asyncio.QueueEmpty:
                    pass

                # Check for client messages
                try:
                    data = await asyncio.wait_for(ws.receive_json(), timeout=0.01)
                    if data.get("type") == "correct" and _live_manager._analyzer:
                        _live_manager._analyzer.scoring_engine.add_point(data["team"])
                    elif data.get("type") == "reassign" and _live_manager._analyzer:
                        _live_manager._analyzer.player_tracker.assign_player(
                            data["track_id"], data["player_id"])
                except asyncio.TimeoutError:
                    pass

                # Send score
                if _live_manager._latest_result:
                    await ws.send_json({
                        "type": "score",
                        "data": _live_manager._latest_result.score,
                    })

            await asyncio.sleep(1 / 30)  # ~30fps
    except WebSocketDisconnect:
        pass
```

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/pipeline/live_manager.py main.py
git commit -m "feat: LiveManager with WebSocket streaming, replay buffer, and camera feed"
```

---

### Task 15: Full Pipeline Integration Test

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 1: Write integration test for the full new pipeline**

Add to `tests/test_integration.py`:

```python
class TestPhase2Pipeline:
    def test_event_detector_full_flow(self, sample_court_corners_pixels):
        """Test: calibration → trackers → event detector → scoring."""
        from cv.court_calibration import CourtCalibration
        from cv.ball_tracker import BallTracker
        from cv.player_tracker import PlayerTracker
        from logic.event_detector import EventDetector
        from logic.scoring_engine import PadelScoringEngine
        from models.config import EventDetectorConfig
        from models.types import ServerInfo, TeamId, MatchState
        import numpy as np

        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)

        config = EventDetectorConfig()
        scoring = PadelScoringEngine(
            golden_point=True,
            first_server=ServerInfo(team_id=TeamId.TEAM_A, player_id="P1"),
            team_players={TeamId.TEAM_A: ["P1", "P2"], TeamId.TEAM_B: ["P3", "P4"]},
        )
        player_tracker = PlayerTracker(cal)
        team_map = {"P1": 1, "P2": 1, "P3": 2, "P4": 2}

        ed = EventDetector(config, cal, scoring, player_tracker, team_map)
        assert ed.state_machine.state == MatchState.IDLE

        # Process some frames with no ball — should stay idle
        events = ed.process(None, [], frame_no=0)
        assert len(events) == 0

    def test_replay_buffer_in_pipeline(self):
        """Test: replay buffer stores and retrieves frames."""
        from pipeline.replay_buffer import ReplayBuffer
        import numpy as np

        rb = ReplayBuffer(max_frames=10)
        for i in range(15):
            frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            rb.add(frame, timestamp=float(i) / 30.0)

        frames = rb.get_frames()
        assert len(frames) == 10
        # First frame should be #5 (oldest after wrap)
        assert abs(frames[0]["timestamp"] - 5.0 / 30.0) < 0.01
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/test_integration.py -v --tb=short`
Expected: All pass (existing 4 + 2 new)

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: ALL tests pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: Phase 2 integration tests for event detector and replay buffer"
```

---

### Task 16: Remove Legacy Endpoint + Final Cleanup

**Files:**
- Modify: `main.py` — remove `/setup` legacy endpoint

- [ ] **Step 1: Remove legacy endpoint**

Remove the `/setup` endpoint from `main.py` (lines 97-104):

```python
# DELETE THIS:
# Legacy endpoint for backwards compatibility
@app.get("/setup")
def get_setup():
    ...
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest tests/ -v --tb=short`
Expected: All tests pass

- [ ] **Step 3: Final commit**

```bash
git add main.py
git commit -m "chore: remove legacy /setup endpoint, Phase 2 pipeline complete"
```
