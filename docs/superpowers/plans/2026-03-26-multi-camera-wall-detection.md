# Multi-Camera Fusion & Wall Hit Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable N-camera fusion with a unified 3D world model and precise wall/fence hit detection for padel match analysis.

**Architecture:** A `PadelCourtModel` defines canonical 3D court geometry (walls as bounded planes). Each camera runs independently as a `CameraNode`, outputting observations. A `WorldFusion` layer merges observations into a single `WorldState` per frame. A `WallCollisionDetector` checks fused trajectories against wall planes. `VideoAnalyzer` and `LiveManager` are refactored to use this pipeline.

**Tech Stack:** Python 3.14, FastAPI, OpenCV (solvePnP, triangulation), NumPy, pytest, threading, queue.Queue

**Spec:** `docs/superpowers/specs/2026-03-26-multi-camera-wall-detection-design.md`

---

## File Structure

**New files:**
- `backend/src/models/court_model.py` — `PadelCourtModel`, `WallSegment` dataclass, standard padel geometry
- `backend/src/cv/camera_node.py` — `CameraNode`, per-camera processing wrapper
- `backend/src/pipeline/world_fusion.py` — `WorldFusion`, multi-camera observation merging + triangulation
- `backend/src/logic/detectors/wall_collision.py` — `WallCollisionDetector`
- `backend/tests/test_court_model.py` — tests for court model + wall geometry
- `backend/tests/test_wall_collision.py` — tests for wall collision detection
- `backend/tests/test_camera_node.py` — tests for camera node
- `backend/tests/test_world_fusion.py` — tests for world fusion

**Modified files:**
- `backend/src/models/types.py` — add `CameraObservation`, `WorldState`, wall hit metadata fields
- `backend/src/models/config.py` — remove `enclosure_bounds`, add court model reference
- `backend/src/cv/camera_model.py` — add `pixel_to_court()` and `court_to_pixel_2d()` adapter methods
- `backend/src/cv/ball_tracker.py` — accept `CameraModel` instead of `CourtCalibration`
- `backend/src/cv/player_tracker.py` — accept `CameraModel` instead of `CourtCalibration`
- `backend/src/logic/detectors/point_end.py` — remove wall-proximity heuristic, consume wall hit events
- `backend/src/logic/event_detector.py` — add `WallCollisionDetector` as sub-detector
- `backend/src/pipeline/video_analyzer.py` — refactor to use `CameraNode` + `WorldFusion`
- `backend/src/pipeline/live_manager.py` — refactor for multi-camera RTSP
- `backend/main.py` — add multi-camera API endpoints
- `backend/tests/conftest.py` — add court model and camera model fixtures

---

## Task 1: PadelCourtModel — 3D Court Geometry

**Files:**
- Create: `backend/src/models/court_model.py`
- Create: `backend/tests/test_court_model.py`

- [ ] **Step 1: Write failing tests for court model**

```python
# backend/tests/test_court_model.py
import pytest
from models.court_model import PadelCourtModel, WallSegment
import numpy as np


def test_default_court_dimensions():
    court = PadelCourtModel()
    assert court.width == 10.0
    assert court.length == 20.0
    assert court.net_y == 10.0


def test_default_wall_segments_count():
    court = PadelCourtModel()
    walls = court.get_wall_segments()
    # 2 back walls (glass) + 4 side glass sections + side fence sections
    assert len(walls) >= 6


def test_back_wall_near():
    court = PadelCourtModel()
    wall = court.get_wall_by_id("back_near_glass")
    assert wall is not None
    assert wall.surface_type == "glass"
    # Back wall near is at Y=0, spans X=0..10, Z=0..4
    assert wall.plane_point[1] == 0.0  # Y=0
    assert wall.bounds_min[2] == 0.0   # Z min
    assert wall.bounds_max[2] == 4.0   # Z max


def test_back_wall_far():
    court = PadelCourtModel()
    wall = court.get_wall_by_id("back_far_glass")
    assert wall is not None
    assert wall.plane_point[1] == 20.0  # Y=20


def test_side_wall_left_glass():
    court = PadelCourtModel()
    wall = court.get_wall_by_id("side_left_glass_near")
    assert wall is not None
    assert wall.surface_type == "glass"
    assert wall.plane_point[0] == 0.0  # X=0


def test_get_bounds():
    court = PadelCourtModel()
    bounds = court.get_bounds()
    assert bounds["x_min"] == 0.0
    assert bounds["x_max"] == 10.0
    assert bounds["y_min"] == 0.0
    assert bounds["y_max"] == 20.0


def test_wall_plane_normal_back_near():
    court = PadelCourtModel()
    wall = court.get_wall_by_id("back_near_glass")
    # Normal should point inward (positive Y)
    assert wall.plane_normal[1] > 0


def test_override_wall_height():
    court = PadelCourtModel(overrides={"back_wall_height": 3.5})
    wall = court.get_wall_by_id("back_near_glass")
    assert wall.bounds_max[2] == 3.5


def test_point_near_wall():
    court = PadelCourtModel()
    # Point at back wall
    assert court.nearest_wall(0.5, 0.1, 1.0) is not None
    # Point in center of court
    assert court.nearest_wall(5.0, 10.0, 0.5) is None


def test_ray_wall_intersection():
    court = PadelCourtModel()
    # Ball moving toward back near wall (Y decreasing)
    p1 = np.array([5.0, 1.0, 1.5])
    p2 = np.array([5.0, -0.1, 1.5])
    hit = court.ray_intersect_walls(p1, p2)
    assert hit is not None
    assert hit["wall_id"] == "back_near_glass"
    assert abs(hit["point"][1] - 0.0) < 0.01  # Hit at Y=0


def test_ray_no_intersection():
    court = PadelCourtModel()
    # Ball moving parallel to walls
    p1 = np.array([5.0, 5.0, 1.0])
    p2 = np.array([5.0, 6.0, 1.0])
    hit = court.ray_intersect_walls(p1, p2)
    assert hit is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_court_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'models.court_model'`

- [ ] **Step 3: Implement PadelCourtModel**

```python
# backend/src/models/court_model.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class WallSegment:
    """A bounded 3D plane representing a wall/fence/glass section."""
    wall_id: str
    surface_type: str  # "glass", "mesh", "fence", "open"
    plane_point: np.ndarray  # A point on the plane (3D)
    plane_normal: np.ndarray  # Inward-facing unit normal (3D)
    bounds_min: np.ndarray  # Min corner of the bounded region (3D)
    bounds_max: np.ndarray  # Max corner of the bounded region (3D)


# Standard padel court dimensions (meters)
COURT_WIDTH = 10.0
COURT_LENGTH = 20.0
NET_Y = 10.0
NET_HEIGHT = 0.88
NET_POST_HEIGHT = 0.92
SERVICE_NEAR_Y = 6.95
SERVICE_FAR_Y = 13.05
SERVICE_CENTER_X = 5.0

# Wall defaults
BACK_WALL_GLASS_HEIGHT = 4.0
BACK_WALL_MESH_HEIGHT = 5.0  # Glass + mesh total
SIDE_GLASS_HEIGHT = 3.0
SIDE_GLASS_DEPTH = 4.0  # How far side glass extends from back wall
SIDE_FENCE_HEIGHT = 4.0


class PadelCourtModel:
    """Canonical 3D geometry of a padel court. Single source of truth."""

    def __init__(self, overrides: Optional[Dict] = None):
        overrides = overrides or {}
        self.width = overrides.get("width", COURT_WIDTH)
        self.length = overrides.get("length", COURT_LENGTH)
        self.net_y = overrides.get("net_y", NET_Y)
        self.service_near_y = overrides.get("service_near_y", SERVICE_NEAR_Y)
        self.service_far_y = overrides.get("service_far_y", SERVICE_FAR_Y)
        self.service_center_x = overrides.get("service_center_x", SERVICE_CENTER_X)

        self._back_wall_height = overrides.get("back_wall_height", BACK_WALL_GLASS_HEIGHT)
        self._back_mesh_height = overrides.get("back_mesh_height", BACK_WALL_MESH_HEIGHT)
        self._side_glass_height = overrides.get("side_glass_height", SIDE_GLASS_HEIGHT)
        self._side_glass_depth = overrides.get("side_glass_depth", SIDE_GLASS_DEPTH)
        self._side_fence_height = overrides.get("side_fence_height", SIDE_FENCE_HEIGHT)

        self._walls: List[WallSegment] = []
        self._build_walls()

    def _build_walls(self):
        """Construct all wall segments from dimensions."""
        W = self.width
        L = self.length
        bh = self._back_wall_height
        mh = self._back_mesh_height
        sgh = self._side_glass_height
        sgd = self._side_glass_depth
        sfh = self._side_fence_height

        # Back wall near (Y=0), glass
        self._walls.append(WallSegment(
            wall_id="back_near_glass", surface_type="glass",
            plane_point=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([0.0, 1.0, 0.0]),
            bounds_min=np.array([0.0, 0.0, 0.0]),
            bounds_max=np.array([W, 0.0, bh]),
        ))
        # Back wall near mesh (above glass)
        if mh > bh:
            self._walls.append(WallSegment(
                wall_id="back_near_mesh", surface_type="mesh",
                plane_point=np.array([0.0, 0.0, bh]),
                plane_normal=np.array([0.0, 1.0, 0.0]),
                bounds_min=np.array([0.0, 0.0, bh]),
                bounds_max=np.array([W, 0.0, mh]),
            ))
        # Back wall far (Y=L), glass
        self._walls.append(WallSegment(
            wall_id="back_far_glass", surface_type="glass",
            plane_point=np.array([0.0, L, 0.0]),
            plane_normal=np.array([0.0, -1.0, 0.0]),
            bounds_min=np.array([0.0, L, 0.0]),
            bounds_max=np.array([W, L, bh]),
        ))
        # Back wall far mesh
        if mh > bh:
            self._walls.append(WallSegment(
                wall_id="back_far_mesh", surface_type="mesh",
                plane_point=np.array([0.0, L, bh]),
                plane_normal=np.array([0.0, -1.0, 0.0]),
                bounds_min=np.array([0.0, L, bh]),
                bounds_max=np.array([W, L, mh]),
            ))
        # Side left glass near (X=0, Y=0..sgd)
        self._walls.append(WallSegment(
            wall_id="side_left_glass_near", surface_type="glass",
            plane_point=np.array([0.0, 0.0, 0.0]),
            plane_normal=np.array([1.0, 0.0, 0.0]),
            bounds_min=np.array([0.0, 0.0, 0.0]),
            bounds_max=np.array([0.0, sgd, sgh]),
        ))
        # Side left glass far (X=0, Y=L-sgd..L)
        self._walls.append(WallSegment(
            wall_id="side_left_glass_far", surface_type="glass",
            plane_point=np.array([0.0, L - sgd, 0.0]),
            plane_normal=np.array([1.0, 0.0, 0.0]),
            bounds_min=np.array([0.0, L - sgd, 0.0]),
            bounds_max=np.array([0.0, L, sgh]),
        ))
        # Side left fence (X=0, Y=sgd..L-sgd)
        self._walls.append(WallSegment(
            wall_id="side_left_fence", surface_type="fence",
            plane_point=np.array([0.0, sgd, 0.0]),
            plane_normal=np.array([1.0, 0.0, 0.0]),
            bounds_min=np.array([0.0, sgd, 0.0]),
            bounds_max=np.array([0.0, L - sgd, sfh]),
        ))
        # Side right glass near (X=W, Y=0..sgd)
        self._walls.append(WallSegment(
            wall_id="side_right_glass_near", surface_type="glass",
            plane_point=np.array([W, 0.0, 0.0]),
            plane_normal=np.array([-1.0, 0.0, 0.0]),
            bounds_min=np.array([W, 0.0, 0.0]),
            bounds_max=np.array([W, sgd, sgh]),
        ))
        # Side right glass far (X=W, Y=L-sgd..L)
        self._walls.append(WallSegment(
            wall_id="side_right_glass_far", surface_type="glass",
            plane_point=np.array([W, L - sgd, 0.0]),
            plane_normal=np.array([-1.0, 0.0, 0.0]),
            bounds_min=np.array([W, L - sgd, 0.0]),
            bounds_max=np.array([W, L, sgh]),
        ))
        # Side right fence (X=W, Y=sgd..L-sgd)
        self._walls.append(WallSegment(
            wall_id="side_right_fence", surface_type="fence",
            plane_point=np.array([W, sgd, 0.0]),
            plane_normal=np.array([-1.0, 0.0, 0.0]),
            bounds_min=np.array([W, sgd, 0.0]),
            bounds_max=np.array([W, L - sgd, sfh]),
        ))

    def get_wall_segments(self) -> List[WallSegment]:
        return list(self._walls)

    def get_wall_by_id(self, wall_id: str) -> Optional[WallSegment]:
        for w in self._walls:
            if w.wall_id == wall_id:
                return w
        return None

    def get_bounds(self) -> Dict[str, float]:
        return {
            "x_min": 0.0, "x_max": self.width,
            "y_min": 0.0, "y_max": self.length,
        }

    def nearest_wall(self, x: float, y: float, z: float,
                     threshold: float = 0.5) -> Optional[WallSegment]:
        """Return the nearest wall if within threshold distance, else None."""
        point = np.array([x, y, z])
        best = None
        best_dist = threshold
        for w in self._walls:
            # Distance from point to plane
            diff = point - w.plane_point
            dist = abs(np.dot(diff, w.plane_normal))
            # Check within bounded region (project onto plane axes)
            if dist < best_dist and self._point_within_wall_bounds(point, w):
                best_dist = dist
                best = w
        return best

    def ray_intersect_walls(self, p1: np.ndarray, p2: np.ndarray
                            ) -> Optional[Dict]:
        """Test if line segment p1→p2 intersects any wall.
        Returns dict with wall_id, surface_type, point, or None."""
        direction = p2 - p1
        best_t = float("inf")
        best_hit = None

        for w in self._walls:
            denom = np.dot(direction, w.plane_normal)
            if abs(denom) < 1e-8:
                continue  # Parallel to plane
            t = np.dot(w.plane_point - p1, w.plane_normal) / denom
            if t < 0 or t > 1:
                continue  # Intersection outside segment
            hit_point = p1 + t * direction
            if self._point_within_wall_bounds(hit_point, w) and t < best_t:
                best_t = t
                best_hit = {
                    "wall_id": w.wall_id,
                    "surface_type": w.surface_type,
                    "point": hit_point,
                    "t": t,
                }
        return best_hit

    @staticmethod
    def _point_within_wall_bounds(point: np.ndarray, wall: WallSegment,
                                  margin: float = 0.1) -> bool:
        """Check if a 3D point is within the wall's bounded rectangle."""
        for i in range(3):
            lo = min(wall.bounds_min[i], wall.bounds_max[i]) - margin
            hi = max(wall.bounds_min[i], wall.bounds_max[i]) + margin
            if point[i] < lo or point[i] > hi:
                return False
        return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_court_model.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/models/court_model.py backend/tests/test_court_model.py
git commit -m "feat: PadelCourtModel with 3D wall geometry and ray intersection"
```

---

## Task 2: Extend Types — CameraObservation, WorldState, Wall Hit Metadata

**Files:**
- Modify: `backend/src/models/types.py`
- Create: `backend/tests/test_world_types.py`

- [ ] **Step 1: Write failing tests for new types**

```python
# backend/tests/test_world_types.py
from models.types import (
    CameraObservation, WorldState, WallHitMetadata,
    BallPosition, PlayerPosition,
)


def test_camera_observation_creation():
    obs = CameraObservation(
        camera_id="cam1",
        ball_pixel=(640, 360),
        ball_bbox=[600, 340, 680, 380],
        ball_court=BallPosition(x=5.0, y=10.0, z=1.0, speed=50.0),
        confidence=0.85,
        player_detections=[],
        timestamp=1.0,
        frame_number=30,
    )
    assert obs.camera_id == "cam1"
    assert obs.confidence == 0.85


def test_camera_observation_no_ball():
    obs = CameraObservation(
        camera_id="cam2",
        ball_pixel=None,
        ball_bbox=None,
        ball_court=None,
        confidence=0.0,
        player_detections=[],
        timestamp=1.0,
        frame_number=30,
    )
    assert obs.ball_court is None


def test_world_state_creation():
    ws = WorldState(
        ball=BallPosition(x=5.0, y=10.0, z=1.5, speed=60.0),
        ball_velocity=(0.0, -5.0, 0.5),
        players=[
            PlayerPosition(player_id="P1", x=3.0, y=5.0),
        ],
        contributing_cameras=["cam1", "cam2"],
        timestamp=1.0,
        frame_number=30,
    )
    assert len(ws.contributing_cameras) == 2
    assert ws.ball.z == 1.5


def test_world_state_no_ball():
    ws = WorldState(
        ball=None,
        ball_velocity=None,
        players=[],
        contributing_cameras=[],
        timestamp=1.0,
        frame_number=30,
    )
    assert ws.ball is None


def test_wall_hit_metadata():
    meta = WallHitMetadata(
        wall_id="back_near_glass",
        surface_type="glass",
        impact_point=(5.0, 0.0, 2.0),
        speed_at_impact=85.0,
        incoming_angle=30.0,
    )
    assert meta.wall_id == "back_near_glass"
    assert meta.speed_at_impact == 85.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_world_types.py -v`
Expected: FAIL — `ImportError: cannot import name 'CameraObservation'`

- [ ] **Step 3: Add new dataclasses to types.py**

Add to the end of `backend/src/models/types.py`:

```python
@dataclass
class CameraObservation:
    """Per-frame observation from a single camera."""
    camera_id: str
    ball_pixel: Optional[tuple] = None       # (px, py) raw pixel center
    ball_bbox: Optional[list] = None         # [x1, y1, x2, y2] raw pixel bbox
    ball_court: Optional[BallPosition] = None  # Projected to court coords
    confidence: float = 0.0
    player_detections: List[Dict] = field(default_factory=list)  # pixel + court
    timestamp: float = 0.0
    frame_number: int = 0


@dataclass
class WorldState:
    """Fused world state from all cameras for a single frame."""
    ball: Optional[BallPosition] = None
    ball_velocity: Optional[tuple] = None  # (vx, vy, vz) in m/s
    players: List[PlayerPosition] = field(default_factory=list)
    contributing_cameras: List[str] = field(default_factory=list)
    timestamp: float = 0.0
    frame_number: int = 0


@dataclass
class WallHitMetadata:
    """Metadata for a wall hit event."""
    wall_id: str = ""
    surface_type: str = ""
    impact_point: tuple = (0.0, 0.0, 0.0)
    speed_at_impact: float = 0.0
    incoming_angle: float = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_world_types.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/models/types.py backend/tests/test_world_types.py
git commit -m "feat: add CameraObservation, WorldState, WallHitMetadata types"
```

---

## Task 3: CameraModel Adapter Methods

**Files:**
- Modify: `backend/src/cv/camera_model.py`
- Modify: `backend/tests/test_court_calibration.py` (add adapter tests)

- [ ] **Step 1: Write failing tests for adapter methods**

Add to the end of existing test file, or create a new section:

```python
# backend/tests/test_camera_model_adapter.py
import pytest
import numpy as np
from cv.camera_model import CameraModel


@pytest.fixture
def calibrated_camera():
    cam = CameraModel()
    # 12 keypoints matching the standard court
    keypoints_2d = [
        [100, 600], [900, 600],   # k1, k2 (near baseline)
        [200, 450], [500, 450], [800, 450],  # k3-k5 (service near)
        [250, 350], [750, 350],   # k6, k7 (net)
        [200, 250], [500, 250], [800, 250],  # k8-k10 (service far)
        [100, 100], [900, 100],   # k11, k12 (far baseline)
    ]
    net_top = [[250, 340], [750, 340]]
    cam.calibrate(keypoints_2d, net_top_2d=net_top,
                  image_width=1000, image_height=700)
    return cam


def test_pixel_to_court_adapter(calibrated_camera):
    """pixel_to_court should delegate to project_to_ground."""
    result = calibrated_camera.pixel_to_court(500, 350)
    assert result is not None
    x, y = result
    # Should be near center court (5, 10)
    assert 3.0 < x < 7.0
    assert 8.0 < y < 12.0


def test_court_to_pixel_2d_adapter(calibrated_camera):
    """court_to_pixel_2d(cx, cy) delegates to court_to_pixel(cx, cy, 0)."""
    result = calibrated_camera.court_to_pixel_2d(5.0, 10.0)
    assert result is not None
    px, py = result
    assert 200 < px < 800
    assert 100 < py < 600


def test_is_in_bounds(calibrated_camera):
    assert calibrated_camera.is_in_bounds(5.0, 10.0) is True
    assert calibrated_camera.is_in_bounds(-5.0, 10.0) is False
    assert calibrated_camera.is_in_bounds(5.0, 25.0) is False


def test_get_court_side(calibrated_camera):
    assert calibrated_camera.get_court_side(5.0, 3.0) == "near"
    assert calibrated_camera.get_court_side(5.0, 15.0) == "far"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_camera_model_adapter.py -v`
Expected: FAIL — `AttributeError: 'CameraModel' object has no attribute 'pixel_to_court'`

- [ ] **Step 3: Add adapter methods to CameraModel**

Add to `backend/src/cv/camera_model.py` in the `CameraModel` class:

```python
# --- Adapter methods (CourtCalibration interface) ---

def pixel_to_court(self, px: float, py: float):
    """Adapter: matches CourtCalibration.pixel_to_court signature."""
    return self.project_to_ground(px, py)

def court_to_pixel_2d(self, cx: float, cy: float):
    """Adapter: 2D court coords to pixel (ground plane)."""
    return self.court_to_pixel(cx, cy, 0.0)

def is_in_bounds(self, x: float, y: float) -> bool:
    return 0.0 <= x <= COURT_WIDTH and 0.0 <= y <= COURT_LENGTH

def get_court_side(self, x: float, y: float) -> str:
    return "near" if y < NET_Y else "far"

def is_in_service_box(self, x: float, y: float, box: str) -> bool:
    boxes = {
        "near_left": (0, 5, 6.95, 10),
        "near_right": (5, 10, 6.95, 10),
        "far_left": (0, 5, 10, 13.05),
        "far_right": (5, 10, 10, 13.05),
    }
    if box not in boxes:
        return False
    x1, x2, y1, y2 = boxes[box]
    return x1 <= x <= x2 and y1 <= y <= y2
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_camera_model_adapter.py -v`
Expected: All PASS

- [ ] **Step 5: Run all existing tests to ensure no regressions**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest -v`
Expected: All existing tests still PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/cv/camera_model.py backend/tests/test_camera_model_adapter.py
git commit -m "feat: add CourtCalibration adapter methods to CameraModel"
```

---

## Task 4: Update BallTracker and PlayerTracker to Accept CameraModel

**Files:**
- Modify: `backend/src/cv/ball_tracker.py`
- Modify: `backend/src/cv/player_tracker.py`
- Modify: `backend/tests/test_ball_tracker.py`
- Modify: `backend/tests/test_player_tracker.py`

- [ ] **Step 1: Write test that BallTracker accepts CameraModel**

Add a test to `backend/tests/test_ball_tracker.py`:

```python
def test_ball_tracker_accepts_camera_model():
    """BallTracker should work with CameraModel via adapter methods."""
    from cv.camera_model import CameraModel
    cam = CameraModel()
    # Even uncalibrated, should construct without error
    tracker = BallTracker(calibration=cam)
    assert tracker is not None
```

- [ ] **Step 2: Run test to check current behavior**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_ball_tracker.py::test_ball_tracker_accepts_camera_model -v`
Check if it passes already (duck typing may make this work). If it fails, proceed to step 3.

- [ ] **Step 3: Update BallTracker to use adapter method names**

In `backend/src/cv/ball_tracker.py`, find any calls to `self.calibration.pixel_to_court(px, py)` — these should already match the adapter. Find any calls to `self.calibration.court_to_pixel(cx, cy)` and update to `self.calibration.court_to_pixel_2d(cx, cy)` if the 2-arg form is used for ground-plane projection.

Also check `PlayerTracker` for the same pattern: it calls `self.calibration.pixel_to_court()` for ground-plane projection from bbox bottom-center.

- [ ] **Step 4: Run all tracker tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_ball_tracker.py tests/test_player_tracker.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/cv/ball_tracker.py backend/src/cv/player_tracker.py backend/tests/test_ball_tracker.py backend/tests/test_player_tracker.py
git commit -m "refactor: BallTracker and PlayerTracker accept CameraModel"
```

---

## Task 5: WallCollisionDetector

**Files:**
- Create: `backend/src/logic/detectors/wall_collision.py`
- Create: `backend/tests/test_wall_collision.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_wall_collision.py
import pytest
import numpy as np
from models.court_model import PadelCourtModel
from logic.detectors.wall_collision import WallCollisionDetector


@pytest.fixture
def court():
    return PadelCourtModel()


@pytest.fixture
def detector(court):
    return WallCollisionDetector(court)


def test_no_hit_in_center(detector):
    """Ball in center of court, no wall hit."""
    result = detector.check(
        ball_pos={"x": 5.0, "y": 10.0, "z": 1.0, "speed": 50.0},
        prev_ball_pos={"x": 5.0, "y": 9.0, "z": 1.0, "speed": 50.0},
    )
    assert result is None


def test_hit_back_near_wall(detector):
    """Ball crosses back near wall (Y=0)."""
    result = detector.check(
        ball_pos={"x": 5.0, "y": -0.1, "z": 1.5, "speed": 60.0},
        prev_ball_pos={"x": 5.0, "y": 0.5, "z": 1.5, "speed": 60.0},
    )
    assert result is not None
    assert result["wall_id"] == "back_near_glass"
    assert result["surface_type"] == "glass"
    assert abs(result["impact_point"][1]) < 0.01


def test_hit_back_far_wall(detector):
    """Ball crosses back far wall (Y=20)."""
    result = detector.check(
        ball_pos={"x": 5.0, "y": 20.1, "z": 2.0, "speed": 70.0},
        prev_ball_pos={"x": 5.0, "y": 19.5, "z": 2.0, "speed": 70.0},
    )
    assert result is not None
    assert result["wall_id"] == "back_far_glass"


def test_hit_side_wall_left(detector):
    """Ball crosses left side wall (X=0) in glass zone."""
    result = detector.check(
        ball_pos={"x": -0.1, "y": 2.0, "z": 1.0, "speed": 40.0},
        prev_ball_pos={"x": 0.5, "y": 2.0, "z": 1.0, "speed": 40.0},
    )
    assert result is not None
    assert "side_left" in result["wall_id"]


def test_hit_side_fence(detector):
    """Ball crosses left side in fence zone (middle of court Y)."""
    result = detector.check(
        ball_pos={"x": -0.1, "y": 10.0, "z": 2.0, "speed": 40.0},
        prev_ball_pos={"x": 0.5, "y": 10.0, "z": 2.0, "speed": 40.0},
    )
    assert result is not None
    assert result["surface_type"] == "fence"


def test_ball_above_wall_no_hit(detector):
    """Ball trajectory above the wall height — no hit."""
    result = detector.check(
        ball_pos={"x": 5.0, "y": -0.1, "z": 6.0, "speed": 60.0},
        prev_ball_pos={"x": 5.0, "y": 0.5, "z": 6.0, "speed": 60.0},
    )
    assert result is None  # Above 5m mesh, ball goes over


def test_speed_at_impact(detector):
    result = detector.check(
        ball_pos={"x": 5.0, "y": -0.1, "z": 1.5, "speed": 85.0},
        prev_ball_pos={"x": 5.0, "y": 0.5, "z": 1.5, "speed": 85.0},
    )
    assert result is not None
    assert result["speed_at_impact"] == 85.0


def test_incoming_angle(detector):
    """Ball hitting wall head-on: angle to normal should be near 0 degrees."""
    result = detector.check(
        ball_pos={"x": 5.0, "y": -0.1, "z": 1.5, "speed": 60.0},
        prev_ball_pos={"x": 5.0, "y": 1.0, "z": 1.5, "speed": 60.0},
    )
    assert result is not None
    # Head-on hit: angle between trajectory and wall normal ≈ 0°
    assert result["incoming_angle"] < 20.0


def test_no_previous_position(detector):
    """First frame, no previous position — no crash."""
    result = detector.check(
        ball_pos={"x": 5.0, "y": 0.0, "z": 1.0, "speed": 50.0},
        prev_ball_pos=None,
    )
    assert result is None


def test_2d_fallback_near_wall(detector):
    """When z is missing (None/0), use 2D proximity fallback."""
    result = detector.check(
        ball_pos={"x": 5.0, "y": 0.2, "z": 0.0, "speed": 50.0},
        prev_ball_pos={"x": 5.0, "y": 1.0, "z": 0.0, "speed": 50.0},
        use_3d=False,
    )
    # 2D fallback: detects proximity to wall at Y=0
    assert result is not None
    assert result["wall_id"] == "back_near_glass"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_wall_collision.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'logic.detectors.wall_collision'`

- [ ] **Step 3: Implement WallCollisionDetector**

```python
# backend/src/logic/detectors/wall_collision.py
from typing import Dict, Optional
import numpy as np
from models.court_model import PadelCourtModel


class WallCollisionDetector:
    """Detects ball-wall collisions using 3D ray-plane intersection."""

    def __init__(self, court_model: PadelCourtModel):
        self._court = court_model
        self._proximity_threshold_2d = 0.3  # meters, for 2D fallback

    def check(self, ball_pos: Optional[Dict],
              prev_ball_pos: Optional[Dict],
              use_3d: bool = True) -> Optional[Dict]:
        """Check if ball trajectory crossed a wall between two positions.

        Returns dict with wall_id, surface_type, impact_point,
        speed_at_impact, incoming_angle — or None.
        """
        if ball_pos is None or prev_ball_pos is None:
            return None

        x, y = ball_pos["x"], ball_pos["y"]
        z = ball_pos.get("z", 0.0) or 0.0
        speed = ball_pos.get("speed", 0.0)

        px, py = prev_ball_pos["x"], prev_ball_pos["y"]
        pz = prev_ball_pos.get("z", 0.0) or 0.0

        if use_3d and (z > 0.01 or pz > 0.01):
            return self._check_3d(
                np.array([px, py, pz]),
                np.array([x, y, z]),
                speed,
            )
        else:
            return self._check_2d(x, y, speed)

    def _check_3d(self, p1: np.ndarray, p2: np.ndarray,
                  speed: float) -> Optional[Dict]:
        hit = self._court.ray_intersect_walls(p1, p2)
        if hit is None:
            return None

        direction = p2 - p1
        wall = self._court.get_wall_by_id(hit["wall_id"])
        angle = self._compute_angle(direction, wall.plane_normal)

        return {
            "wall_id": hit["wall_id"],
            "surface_type": hit["surface_type"],
            "impact_point": tuple(hit["point"].tolist()),
            "speed_at_impact": speed,
            "incoming_angle": angle,
        }

    def _check_2d(self, x: float, y: float,
                  speed: float) -> Optional[Dict]:
        """Fallback: 2D proximity check against nearest wall."""
        wall = self._court.nearest_wall(x, y, 0.5,
                                        threshold=self._proximity_threshold_2d)
        if wall is None:
            return None
        return {
            "wall_id": wall.wall_id,
            "surface_type": wall.surface_type,
            "impact_point": (x, y, 0.0),
            "speed_at_impact": speed,
            "incoming_angle": 0.0,  # Unknown without 3D
        }

    @staticmethod
    def _compute_angle(direction: np.ndarray,
                       normal: np.ndarray) -> float:
        """Compute angle between trajectory and wall normal in degrees."""
        d_norm = np.linalg.norm(direction)
        n_norm = np.linalg.norm(normal)
        if d_norm < 1e-8 or n_norm < 1e-8:
            return 0.0
        cos_angle = abs(np.dot(direction, normal) / (d_norm * n_norm))
        cos_angle = np.clip(cos_angle, 0.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def reset(self):
        """Reset state between points (stateless, but matches detector interface)."""
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_wall_collision.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/logic/detectors/wall_collision.py backend/tests/test_wall_collision.py
git commit -m "feat: WallCollisionDetector with 3D ray-plane intersection"
```

---

## Task 6: Integrate WallCollisionDetector into EventDetector

**Files:**
- Modify: `backend/src/logic/event_detector.py`
- Modify: `backend/src/logic/detectors/point_end.py`
- Modify: `backend/src/models/config.py`
- Modify: `backend/tests/test_event_detector.py`
- Modify: `backend/tests/test_point_end.py`

- [ ] **Step 1: Write failing test for wall hit event emission**

Add to `backend/tests/test_event_detector.py`:

```python
def test_wall_hit_event_emitted(mock_calibration, mock_scoring, mock_player_tracker):
    """EventDetector should emit WALL_HIT when ball crosses a wall."""
    from models.court_model import PadelCourtModel
    config = EventDetectorConfig()
    court = PadelCourtModel()
    team_map = {"P1": 1, "P2": 1, "P3": 2, "P4": 2}
    ed = EventDetector(config, mock_calibration, mock_scoring,
                       mock_player_tracker, team_map, court_model=court)

    # Put state machine in RALLY
    ed.state_machine.on_serve_started()
    ed.state_machine.on_serve_result(True)

    # Simulate ball hitting back near wall
    events = ed.process(
        ball_pos={"x": 5.0, "y": -0.1, "z": 1.5, "speed": 60.0},
        player_positions=[],
        frame_no=100,
    )
    wall_events = [e for e in events if e.event_type.value == "WALL_HIT"]
    assert len(wall_events) >= 1
    assert "wall_id" in wall_events[0].metadata
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_event_detector.py::test_wall_hit_event_emitted -v`
Expected: FAIL

- [ ] **Step 3: Add court_model parameter to EventDetector and wire WallCollisionDetector**

In `backend/src/logic/event_detector.py`:

1. Add import: `from logic.detectors.wall_collision import WallCollisionDetector`
2. Add import: `from models.court_model import PadelCourtModel`
3. Add `court_model: PadelCourtModel = None` parameter to `__init__`
4. In `__init__`: `self._wall_detector = WallCollisionDetector(court_model) if court_model else None`
5. Store `self._prev_ball_pos = None`
6. In `process()`, during RALLY state, before point_end check:
   ```python
   if self._wall_detector:
       wall_hit = self._wall_detector.check(ball_pos, self._prev_ball_pos)
       if wall_hit:
           events.append(MatchEvent(
               event_type=EventType.WALL_HIT,
               timestamp=frame_no / 30.0,
               frame_number=frame_no,
               position=CourtPoint(x=ball_pos["x"], y=ball_pos["y"]),
               metadata=wall_hit,
           ))
   self._prev_ball_pos = ball_pos
   ```

- [ ] **Step 4: Update PointEndDetector to use wall hit events instead of heuristic**

In `backend/src/logic/detectors/point_end.py`:

1. Add `court_model` parameter to `__init__` (optional, defaults to None). If provided, use `court_model.get_bounds()` for out-of-bounds checks instead of `config.enclosure_bounds`. If not provided, fall back to config bounds for backward compat.
   ```python
   def __init__(self, config: EventDetectorConfig, court_model=None):
       self._config = config
       if court_model:
           b = court_model.get_bounds()
           self._bounds = {"x_min": b["x_min"] - 0.5, "x_max": b["x_max"] + 0.5,
                           "y_min": b["y_min"] - 1.0, "y_max": b["y_max"] + 1.0}
       else:
           self._bounds = config.enclosure_bounds
   ```
2. Remove the wall-proximity check block (lines 35-44: the section checking ball near wall margins at Z > 0.5)
3. Add `wall_hit=None` keyword parameter to `check()`, keeping existing arg order:
   ```python
   def check(self, bounce, ball_pos, ball_lost, wall_hit=None):
   ```
4. Add wall-before-bounce logic using the wall_hit dict (after the out-of-bounds check, replacing the removed block):
   ```python
   # 3. Wall before bounce (from WallCollisionDetector)
   if wall_hit and self._had_bounce:
       ball_side = "near" if y < NET_Y else "far"
       if self._bounces_per_side[ball_side] == 0:
           return {"reason": PointReason.WALL_BEFORE_BOUNCE, "side": ball_side}
   ```

- [ ] **Step 5: Update EventDetector to pass court_model and wall_hit to PointEndDetector**

In `backend/src/logic/event_detector.py`:

1. Pass `court_model` to `PointEndDetector` constructor:
   ```python
   self._point_end_detector = PointEndDetector(config, court_model=court_model)
   ```
2. In the RALLY section, pass the wall_hit result to `PointEndDetector.check()` (keeping existing arg order):
   ```python
   point_end = self._point_end_detector.check(bounce, ball_pos, ball_lost,
                                               wall_hit=wall_hit)
   ```
   Where `wall_hit` comes from the `WallCollisionDetector.check()` call earlier in the same frame.

**Note:** `enclosure_bounds` stays in `EventDetectorConfig` for backward compatibility — it's only unused when `court_model` is provided.

- [ ] **Step 7: Run all event detector and point end tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_event_detector.py tests/test_point_end.py -v`
Expected: All PASS (may need to update existing point_end tests that relied on wall-proximity heuristic)

- [ ] **Step 8: Run full test suite**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest -v`
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/logic/event_detector.py backend/src/logic/detectors/point_end.py backend/src/models/config.py backend/tests/test_event_detector.py backend/tests/test_point_end.py
git commit -m "feat: integrate WallCollisionDetector into EventDetector pipeline"
```

---

## Task 7: CameraNode — Per-Camera Processing Wrapper

**Files:**
- Create: `backend/src/cv/camera_node.py`
- Create: `backend/tests/test_camera_node.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_camera_node.py
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from cv.camera_node import CameraNode, CameraHealth


def test_camera_node_creation():
    node = CameraNode(camera_id="cam1", label="Center High")
    assert node.camera_id == "cam1"
    assert node.health == CameraHealth.UNCALIBRATED


def test_camera_node_calibrate():
    node = CameraNode(camera_id="cam1", label="Test")
    keypoints = [
        [100, 600], [900, 600],
        [200, 450], [500, 450], [800, 450],
        [250, 350], [750, 350],
        [200, 250], [500, 250], [800, 250],
        [100, 100], [900, 100],
    ]
    node.calibrate(keypoints, image_width=1000, image_height=700)
    assert node.health == CameraHealth.ACTIVE
    assert node.camera_model is not None


def test_process_frame_returns_observation():
    node = CameraNode(camera_id="cam1", label="Test",
                      detector_type="yolo")
    # Mock detectors to avoid loading real models
    node._ball_detector = MagicMock()
    node._ball_detector.detect.return_value = [400, 300, 420, 320]
    node._player_detector = MagicMock()
    node._player_detector.detect.return_value = np.empty((0, 6))
    node._camera_model = MagicMock()
    node._camera_model.project_to_ground.return_value = (5.0, 10.0)
    node._camera_model.has_3d.return_value = False
    node._calibrated = True

    frame = np.zeros((700, 1000, 3), dtype=np.uint8)
    obs = node.process_frame(frame, frame_number=1, timestamp=0.033)

    assert obs.camera_id == "cam1"
    assert obs.ball_pixel is not None
    assert obs.ball_court is not None


def test_process_frame_no_ball():
    node = CameraNode(camera_id="cam1", label="Test")
    node._ball_detector = MagicMock()
    node._ball_detector.detect.return_value = None
    node._player_detector = MagicMock()
    node._player_detector.detect.return_value = np.empty((0, 6))
    node._calibrated = True
    node._camera_model = MagicMock()

    frame = np.zeros((700, 1000, 3), dtype=np.uint8)
    obs = node.process_frame(frame, frame_number=1, timestamp=0.033)
    assert obs.ball_pixel is None
    assert obs.ball_court is None


def test_health_degrades_on_no_frames():
    node = CameraNode(camera_id="cam1", label="Test")
    node._calibrated = True
    node._health = CameraHealth.ACTIVE
    node._last_frame_time = 0.0
    # Simulate 3 seconds with no frames
    node.update_health(current_time=3.0)
    assert node.health == CameraHealth.DEGRADED


def test_health_disconnects_after_long_gap():
    node = CameraNode(camera_id="cam1", label="Test")
    node._calibrated = True
    node._health = CameraHealth.ACTIVE
    node._last_frame_time = 0.0
    node.update_health(current_time=11.0)
    assert node.health == CameraHealth.DISCONNECTED


def test_reprojection_error_quality():
    node = CameraNode(camera_id="cam1", label="Test")
    node._reprojection_error = 3.0
    assert node.quality_weight() == 1.0  # <5px = good

    node._reprojection_error = 10.0
    assert node.quality_weight() == 0.5  # 5-15px = acceptable

    node._reprojection_error = 20.0
    assert node.quality_weight() == 0.0  # >15px = excluded
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_camera_node.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement CameraNode**

```python
# backend/src/cv/camera_node.py
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List
import numpy as np

from cv.camera_model import CameraModel
from models.types import CameraObservation, BallPosition


class CameraHealth(Enum):
    UNCALIBRATED = "uncalibrated"
    ACTIVE = "active"
    DEGRADED = "degraded"
    DISCONNECTED = "disconnected"


class CameraNode:
    """Per-camera processing wrapper. Runs detection and projects to court coords."""

    DEGRADED_TIMEOUT = 2.0   # seconds with no frames → degraded
    DISCONNECT_TIMEOUT = 10.0  # seconds → disconnected

    def __init__(self, camera_id: str, label: str = "",
                 detector_type: str = "yolo"):
        self.camera_id = camera_id
        self.label = label
        self._detector_type = detector_type
        self._camera_model = CameraModel()
        self._ball_detector = None
        self._player_detector = None
        self._calibrated = False
        self._health = CameraHealth.UNCALIBRATED
        self._last_frame_time = 0.0
        self._reprojection_error = float("inf")

    @property
    def health(self) -> CameraHealth:
        return self._health

    @property
    def camera_model(self) -> CameraModel:
        return self._camera_model

    def calibrate(self, keypoints_2d, net_top_2d=None,
                  image_width=1280, image_height=720):
        """Calibrate camera against court model."""
        self._camera_model.calibrate(
            keypoints_2d=keypoints_2d,
            net_top_2d=net_top_2d,
            image_width=image_width,
            image_height=image_height,
        )
        self._calibrated = True
        self._health = CameraHealth.ACTIVE
        # TODO: compute reprojection error

    def init_detectors(self):
        """Initialize detection models. Call after calibration."""
        from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector, YoloPlayerDetector
        from cv.detectors.fast_ball import FastBallDetector

        unified = UnifiedYoloDetector()
        self._player_detector = YoloPlayerDetector(unified)

        if self._detector_type == "fast":
            self._ball_detector = FastBallDetector(
                yolo_fallback=YoloBallDetector(unified))
        elif self._detector_type == "tracknet":
            from cv.detectors.tracknet import TrackNetBallDetector
            self._ball_detector = TrackNetBallDetector(
                yolo_fallback=YoloBallDetector(unified))
        else:
            self._ball_detector = YoloBallDetector(unified)

    def process_frame(self, frame: np.ndarray, frame_number: int,
                      timestamp: float) -> CameraObservation:
        """Run detection on a frame and return observation in court coords."""
        self._last_frame_time = timestamp

        ball_pixel = None
        ball_bbox = None
        ball_court = None
        confidence = 0.0

        # Ball detection
        if self._ball_detector:
            bbox = self._ball_detector.detect(frame, frame_number)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                ball_pixel = (cx, cy)
                ball_bbox = list(bbox)
                confidence = 0.8  # TODO: get from detector

                if self._calibrated:
                    court_xy = self._camera_model.project_to_ground(cx, cy)
                    if court_xy:
                        ball_court = BallPosition(
                            x=court_xy[0], y=court_xy[1],
                            timestamp=timestamp,
                        )

        # Player detection
        player_detections = []
        if self._player_detector:
            dets = self._player_detector.detect(frame, frame_number)
            for det in dets:
                x1, y1, x2, y2 = det[:4]
                foot_x, foot_y = (x1 + x2) / 2, y2
                court_pos = None
                if self._calibrated:
                    court_pos = self._camera_model.project_to_ground(
                        foot_x, foot_y)
                player_detections.append({
                    "bbox": list(det[:4]),
                    "pixel": (foot_x, foot_y),
                    "court": court_pos,
                    "confidence": float(det[4]) if len(det) > 4 else 0.0,
                })

        return CameraObservation(
            camera_id=self.camera_id,
            ball_pixel=ball_pixel,
            ball_bbox=ball_bbox,
            ball_court=ball_court,
            confidence=confidence,
            player_detections=player_detections,
            timestamp=timestamp,
            frame_number=frame_number,
        )

    def update_health(self, current_time: float):
        """Update health based on frame recency."""
        if not self._calibrated:
            return
        gap = current_time - self._last_frame_time
        if gap > self.DISCONNECT_TIMEOUT:
            self._health = CameraHealth.DISCONNECTED
        elif gap > self.DEGRADED_TIMEOUT:
            self._health = CameraHealth.DEGRADED
        else:
            if self._reprojection_error <= 15.0:
                self._health = CameraHealth.ACTIVE
            else:
                self._health = CameraHealth.DEGRADED

    def quality_weight(self) -> float:
        """Weight for fusion: 0.0 (excluded) to 1.0 (full)."""
        if self._health in (CameraHealth.DISCONNECTED,
                            CameraHealth.UNCALIBRATED):
            return 0.0
        if self._reprojection_error > 15.0:
            return 0.0
        if self._reprojection_error > 5.0:
            return 0.5
        return 1.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_camera_node.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/cv/camera_node.py backend/tests/test_camera_node.py
git commit -m "feat: CameraNode per-camera processing with health tracking"
```

---

## Task 8: WorldFusion — Multi-Camera Observation Merging

**Files:**
- Create: `backend/src/pipeline/world_fusion.py`
- Create: `backend/tests/test_world_fusion.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_world_fusion.py
import pytest
from unittest.mock import MagicMock
from pipeline.world_fusion import WorldFusion
from models.types import CameraObservation, BallPosition, WorldState
from models.court_model import PadelCourtModel
from cv.camera_node import CameraNode, CameraHealth


@pytest.fixture
def court():
    return PadelCourtModel()


@pytest.fixture
def fusion(court):
    return WorldFusion(court_model=court)


def _make_obs(camera_id, x, y, z=0.0, speed=0.0, conf=0.8,
              pixel=(500, 300), frame=1, ts=0.033):
    return CameraObservation(
        camera_id=camera_id,
        ball_pixel=pixel,
        ball_bbox=[490, 290, 510, 310],
        ball_court=BallPosition(x=x, y=y, z=z, speed=speed, timestamp=ts),
        confidence=conf,
        player_detections=[],
        timestamp=ts,
        frame_number=frame,
    )


def test_single_camera_fusion(fusion):
    obs = [_make_obs("cam1", x=5.0, y=10.0, z=1.0, speed=50.0)]
    weights = {"cam1": 1.0}
    ws = fusion.fuse(obs, weights)
    assert ws.ball is not None
    assert abs(ws.ball.x - 5.0) < 0.01
    assert ws.contributing_cameras == ["cam1"]


def test_two_camera_weighted_average(fusion):
    obs = [
        _make_obs("cam1", x=4.0, y=10.0, z=1.0, conf=0.9),
        _make_obs("cam2", x=6.0, y=10.0, z=1.5, conf=0.9),
    ]
    weights = {"cam1": 1.0, "cam2": 1.0}
    ws = fusion.fuse(obs, weights)
    assert ws.ball is not None
    # Weighted average of x=4 and x=6 with equal weights → ~5.0
    assert 4.5 < ws.ball.x < 5.5


def test_no_observations_returns_prediction(fusion):
    # First, feed one observation to establish state
    obs1 = [_make_obs("cam1", x=5.0, y=10.0, z=1.0, speed=50.0,
                       frame=1, ts=0.033)]
    fusion.fuse(obs1, {"cam1": 1.0})

    # Then fuse with no observations
    ws = fusion.fuse([], {})
    # Should return Kalman prediction (not None)
    assert ws.ball is not None


def test_zero_cameras_first_frame(fusion):
    ws = fusion.fuse([], {})
    assert ws.ball is None
    assert ws.contributing_cameras == []


def test_no_ball_observations(fusion):
    obs = [CameraObservation(
        camera_id="cam1", ball_pixel=None, ball_bbox=None,
        ball_court=None, confidence=0.0, player_detections=[],
        timestamp=0.033, frame_number=1,
    )]
    ws = fusion.fuse(obs, {"cam1": 1.0})
    assert ws.ball is None


def test_player_fusion_deduplication(fusion):
    obs = [
        CameraObservation(
            camera_id="cam1", ball_pixel=None, ball_bbox=None,
            ball_court=None, confidence=0.0,
            player_detections=[
                {"court": (3.0, 5.0), "bbox": [100, 200, 150, 400],
                 "confidence": 0.9, "pixel": (125, 400)},
            ],
            timestamp=0.033, frame_number=1,
        ),
        CameraObservation(
            camera_id="cam2", ball_pixel=None, ball_bbox=None,
            ball_court=None, confidence=0.0,
            player_detections=[
                # Same player, slightly different position
                {"court": (3.1, 5.1), "bbox": [200, 210, 260, 410],
                 "confidence": 0.85, "pixel": (230, 410)},
            ],
            timestamp=0.033, frame_number=1,
        ),
    ]
    ws = fusion.fuse(obs, {"cam1": 1.0, "cam2": 1.0})
    # Should deduplicate to 1 player, not 2
    assert len(ws.players) == 1


def test_velocity_computation(fusion):
    obs1 = [_make_obs("cam1", x=5.0, y=10.0, z=1.0, frame=1, ts=0.033)]
    fusion.fuse(obs1, {"cam1": 1.0})

    obs2 = [_make_obs("cam1", x=5.0, y=9.0, z=1.0, frame=2, ts=0.066)]
    ws = fusion.fuse(obs2, {"cam1": 1.0})
    assert ws.ball_velocity is not None
    # Ball moved -1.0 in Y over 0.033s
    vy = ws.ball_velocity[1]
    assert vy < 0  # Moving in -Y direction
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_world_fusion.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement WorldFusion**

```python
# backend/src/pipeline/world_fusion.py
from typing import Dict, List, Optional, Tuple
import numpy as np

from models.types import (
    CameraObservation, WorldState, BallPosition, PlayerPosition,
)
from models.court_model import PadelCourtModel


# Proximity threshold for player deduplication (meters)
PLAYER_DEDUP_DISTANCE = 2.0


class WorldFusion:
    """Merges observations from N cameras into a single WorldState per frame."""

    def __init__(self, court_model: PadelCourtModel):
        self._court = court_model
        self._prev_ball: Optional[BallPosition] = None
        self._prev_time: float = 0.0

    def fuse(self, observations: List[CameraObservation],
             camera_weights: Dict[str, float]) -> WorldState:
        """Fuse observations from all cameras for one frame."""
        # Separate ball observations from no-ball
        ball_obs = [
            o for o in observations
            if o.ball_court is not None and camera_weights.get(o.camera_id, 0) > 0
        ]
        timestamp = observations[0].timestamp if observations else self._prev_time
        frame_number = observations[0].frame_number if observations else 0

        # Ball fusion
        ball, contributing = self._fuse_ball(ball_obs, camera_weights)

        # Velocity
        velocity = None
        if ball and self._prev_ball:
            dt = timestamp - self._prev_time
            if dt > 0:
                vx = (ball.x - self._prev_ball.x) / dt
                vy = (ball.y - self._prev_ball.y) / dt
                vz = (ball.z - self._prev_ball.z) / dt
                velocity = (vx, vy, vz)
                speed = np.sqrt(vx**2 + vy**2 + vz**2) * 3.6  # m/s → km/h
                ball = BallPosition(
                    x=ball.x, y=ball.y, z=ball.z,
                    speed=speed, timestamp=timestamp,
                )

        # Player fusion
        players = self._fuse_players(observations, camera_weights)

        # Update state
        if ball:
            self._prev_ball = ball
        self._prev_time = timestamp

        return WorldState(
            ball=ball,
            ball_velocity=velocity,
            players=players,
            contributing_cameras=contributing,
            timestamp=timestamp,
            frame_number=frame_number,
        )

    def _fuse_ball(self, ball_obs: List[CameraObservation],
                   weights: Dict[str, float]
                   ) -> Tuple[Optional[BallPosition], List[str]]:
        """Weighted average of ball positions."""
        if not ball_obs:
            # Predict from last known
            if self._prev_ball:
                return self._prev_ball, []
            return None, []

        if len(ball_obs) == 1:
            o = ball_obs[0]
            return o.ball_court, [o.camera_id]

        # Weighted average
        total_w = 0.0
        wx, wy, wz = 0.0, 0.0, 0.0
        cameras = []
        for o in ball_obs:
            w = weights.get(o.camera_id, 0.0) * o.confidence
            if w <= 0:
                continue
            wx += o.ball_court.x * w
            wy += o.ball_court.y * w
            wz += o.ball_court.z * w
            total_w += w
            cameras.append(o.camera_id)

        if total_w <= 0:
            return None, []

        ball = BallPosition(
            x=wx / total_w,
            y=wy / total_w,
            z=wz / total_w,
            speed=ball_obs[0].ball_court.speed,
            timestamp=ball_obs[0].timestamp,
        )
        return ball, cameras

    def _fuse_players(self, observations: List[CameraObservation],
                      weights: Dict[str, float]
                      ) -> List[PlayerPosition]:
        """Merge and deduplicate player detections across cameras."""
        all_players = []
        for obs in observations:
            w = weights.get(obs.camera_id, 0.0)
            if w <= 0:
                continue
            for det in obs.player_detections:
                if det.get("court") is None:
                    continue
                cx, cy = det["court"]
                all_players.append({
                    "x": cx, "y": cy,
                    "confidence": det.get("confidence", 0.0) * w,
                    "camera_id": obs.camera_id,
                })

        # Deduplicate by proximity
        merged = []
        used = set()
        for i, p in enumerate(all_players):
            if i in used:
                continue
            group = [p]
            used.add(i)
            for j in range(i + 1, len(all_players)):
                if j in used:
                    continue
                dist = np.sqrt((p["x"] - all_players[j]["x"])**2 +
                               (p["y"] - all_players[j]["y"])**2)
                if dist < PLAYER_DEDUP_DISTANCE:
                    group.append(all_players[j])
                    used.add(j)
            # Average position, weighted by confidence
            total_c = sum(g["confidence"] for g in group)
            if total_c > 0:
                avg_x = sum(g["x"] * g["confidence"] for g in group) / total_c
                avg_y = sum(g["y"] * g["confidence"] for g in group) / total_c
            else:
                avg_x = sum(g["x"] for g in group) / len(group)
                avg_y = sum(g["y"] for g in group) / len(group)
            merged.append(PlayerPosition(
                player_id="", x=avg_x, y=avg_y,
            ))
        return merged

    def reset(self):
        self._prev_ball = None
        self._prev_time = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_world_fusion.py -v`
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/pipeline/world_fusion.py backend/tests/test_world_fusion.py
git commit -m "feat: WorldFusion multi-camera observation merging"
```

---

## Task 9: Refactor VideoAnalyzer to Use CameraNode + WorldFusion

**Files:**
- Modify: `backend/src/pipeline/video_analyzer.py`
- Modify: `backend/tests/test_video_analyzer.py`

- [ ] **Step 1: Write test for single-camera backward compatibility**

```python
# Add to backend/tests/test_video_analyzer.py
def test_video_analyzer_single_camera_compat():
    """VideoAnalyzer with one camera should behave like the old single-camera mode."""
    from unittest.mock import MagicMock
    from pipeline.video_analyzer import VideoAnalyzer
    from models.config import EventDetectorConfig

    cal = MagicMock()
    cal.pixel_to_court.return_value = (5.0, 10.0)
    cal.project_to_ground.return_value = (5.0, 10.0)
    cal.has_3d.return_value = False
    cal.court_to_pixel.return_value = (500, 350)
    cal.court_to_pixel_2d.return_value = (500, 350)
    cal.is_in_bounds.return_value = True
    cal.get_court_side.return_value = "near"
    cal.is_in_service_box.return_value = False

    config = EventDetectorConfig()
    analyzer = VideoAnalyzer(
        match_id="test",
        calibration=cal,
        config=config,
        detector_type="yolo",
    )
    # Should have created internal CameraNode + WorldFusion
    assert analyzer._world_fusion is not None
```

- [ ] **Step 2: Refactor VideoAnalyzer internals**

Key changes to `backend/src/pipeline/video_analyzer.py`:

1. Import `CameraNode`, `WorldFusion`, `PadelCourtModel`
2. In `__init__`:
   - Create `PadelCourtModel()`
   - Create a single `CameraNode` from the existing detectors
   - Create `WorldFusion(court_model)`
   - Pass `court_model` to `EventDetector`
3. In `process_frame`:
   - Call `CameraNode.process_frame()` to get `CameraObservation`
   - Call `WorldFusion.fuse([observation], weights)` to get `WorldState`
   - Convert `WorldState` to the existing dict format for `EventDetector.process()`
   - Rest of the pipeline (player tracking, scoring) stays the same
4. Add `add_camera(camera_node)` method for multi-camera support

Keep all existing public methods and return types unchanged.

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest -v`
Expected: All PASS — single-camera behavior unchanged

- [ ] **Step 4: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/pipeline/video_analyzer.py backend/tests/test_video_analyzer.py
git commit -m "refactor: VideoAnalyzer uses CameraNode + WorldFusion internally"
```

---

## Task 10: Refactor LiveManager for Multi-Camera RTSP

**Files:**
- Modify: `backend/src/pipeline/live_manager.py`
- Modify: `backend/tests/test_integration.py`

- [ ] **Step 1: Write test for multi-camera live manager**

```python
# Add to a new test or existing integration tests
def test_live_manager_multi_camera():
    from unittest.mock import MagicMock, patch
    from pipeline.live_manager import LiveManager
    from cv.camera_node import CameraNode

    cam1 = MagicMock(spec=CameraNode)
    cam1.camera_id = "cam1"
    cam1.quality_weight.return_value = 1.0

    cam2 = MagicMock(spec=CameraNode)
    cam2.camera_id = "cam2"
    cam2.quality_weight.return_value = 1.0

    manager = LiveManager(
        camera_nodes=[cam1, cam2],
        court_model=MagicMock(),
    )
    assert len(manager._camera_nodes) == 2
```

- [ ] **Step 2: Refactor LiveManager**

Key changes to `backend/src/pipeline/live_manager.py`:

1. Constructor accepts `camera_nodes: List[CameraNode]` and `court_model: PadelCourtModel` instead of a single `VideoAnalyzer`
2. Each `CameraNode` gets its own capture thread (device_id or RTSP URL stored per node)
3. Create `WorldFusion(court_model)` internally
4. Main loop: collect observations from all camera threads → fuse → run event detection
5. Keep existing WebSocket streaming, replay buffer, JPEG encoding from the primary camera
6. Backward compat: if constructed with a single analyzer (old API), wrap it in a CameraNode internally

- [ ] **Step 3: Run tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/src/pipeline/live_manager.py
git commit -m "refactor: LiveManager supports multiple CameraNode RTSP feeds"
```

---

## Task 11: Multi-Camera API Endpoints

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_api.py`

- [ ] **Step 1: Write failing API tests**

```python
# Add to backend/tests/test_api.py
def test_add_camera_to_match(client, match_id):
    resp = client.post(f"/match/{match_id}/cameras", json={
        "camera_id": "cam1",
        "label": "Center High",
        "source_type": "rtsp",
        "source_path": "rtsp://192.168.1.100/stream",
    })
    assert resp.status_code == 200
    assert resp.json()["camera_id"] == "cam1"


def test_calibrate_camera(client, match_id):
    # Add camera first
    client.post(f"/match/{match_id}/cameras", json={
        "camera_id": "cam1", "label": "Test",
    })
    keypoints = [[100+i*70, 600-i*40] for i in range(12)]
    resp = client.post(f"/match/{match_id}/cameras/cam1/calibrate", json={
        "corners": keypoints,
        "image_width": 1280,
        "image_height": 720,
    })
    assert resp.status_code == 200


def test_set_court_model_overrides(client, match_id):
    resp = client.post(f"/match/{match_id}/court-model", json={
        "back_wall_height": 3.5,
        "side_glass_height": 2.5,
    })
    assert resp.status_code == 200


def test_list_cameras(client, match_id):
    client.post(f"/match/{match_id}/cameras", json={
        "camera_id": "cam1", "label": "Cam 1",
    })
    client.post(f"/match/{match_id}/cameras", json={
        "camera_id": "cam2", "label": "Cam 2",
    })
    resp = client.get(f"/match/{match_id}")
    data = resp.json()
    assert len(data.get("cameras", [])) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_api.py -v -k camera`
Expected: FAIL — 404 or endpoint not found

- [ ] **Step 3: Add endpoints to main.py**

Add to `backend/main.py`:

1. `POST /match/{id}/cameras` — add a camera entry to match config
2. `POST /match/{id}/cameras/{cam_id}/calibrate` — calibrate one camera via `CameraModel`
3. `POST /match/{id}/court-model` — set wall reference point overrides in match config
4. Update `POST /match/{id}/analyze` to create CameraNodes from the cameras list
5. Update `POST /live/start` to accept multiple RTSP URLs

Pydantic models:
```python
class AddCameraRequest(BaseModel):
    camera_id: str
    label: str = ""
    source_type: str = "file"  # "file" or "rtsp"
    source_path: str = ""

class CourtModelOverrideRequest(BaseModel):
    back_wall_height: Optional[float] = None
    side_glass_height: Optional[float] = None
    side_glass_depth: Optional[float] = None
    side_fence_height: Optional[float] = None
```

- [ ] **Step 4: Run API tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_api.py -v`
Expected: All PASS

- [ ] **Step 5: Run full test suite**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/main.py backend/tests/test_api.py
git commit -m "feat: multi-camera API endpoints for camera management and court model"
```

---

## Task 12: Update conftest.py with Shared Fixtures

**Files:**
- Modify: `backend/tests/conftest.py`

- [ ] **Step 1: Add court model and camera fixtures**

```python
# Add to backend/tests/conftest.py
from models.court_model import PadelCourtModel

@pytest.fixture
def court_model():
    return PadelCourtModel()

@pytest.fixture
def calibrated_camera_model():
    from cv.camera_model import CameraModel
    cam = CameraModel()
    keypoints = [
        [100, 600], [900, 600],
        [200, 450], [500, 450], [800, 450],
        [250, 350], [750, 350],
        [200, 250], [500, 250], [800, 250],
        [100, 100], [900, 100],
    ]
    cam.calibrate(keypoints, image_width=1000, image_height=700)
    return cam
```

- [ ] **Step 2: Run full test suite to ensure fixtures work**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/tests/conftest.py
git commit -m "test: add court model and camera model shared fixtures"
```

---

## Task 13: End-to-End Integration Test

**Files:**
- Create: `backend/tests/test_multi_camera_integration.py`

- [ ] **Step 1: Write integration test**

```python
# backend/tests/test_multi_camera_integration.py
"""End-to-end test: multi-camera fusion with wall hit detection."""
import pytest
import numpy as np
from unittest.mock import MagicMock

from models.court_model import PadelCourtModel
from models.config import EventDetectorConfig
from models.types import EventType, BallPosition, CameraObservation
from cv.camera_node import CameraNode
from pipeline.world_fusion import WorldFusion
from logic.detectors.wall_collision import WallCollisionDetector


@pytest.fixture
def court():
    return PadelCourtModel()


def test_wall_hit_detected_through_fusion(court):
    """Ball detected by two cameras, trajectory crosses back wall."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    # Frame 1: ball at (5, 2, 1.5)
    obs1 = [
        CameraObservation(
            camera_id="cam1", ball_pixel=(500, 400), ball_bbox=[490, 390, 510, 410],
            ball_court=BallPosition(x=5.0, y=2.0, z=1.5, speed=70.0),
            confidence=0.9, player_detections=[], timestamp=0.033, frame_number=1,
        ),
        CameraObservation(
            camera_id="cam2", ball_pixel=(480, 410), ball_bbox=[470, 400, 490, 420],
            ball_court=BallPosition(x=5.1, y=1.9, z=1.4, speed=68.0),
            confidence=0.85, player_detections=[], timestamp=0.033, frame_number=1,
        ),
    ]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0, "cam2": 1.0})
    prev_pos = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}

    # Frame 2: ball at (5, -0.1, 1.5) — crossed back wall
    obs2 = [
        CameraObservation(
            camera_id="cam1", ball_pixel=(500, 500), ball_bbox=[490, 490, 510, 510],
            ball_court=BallPosition(x=5.0, y=-0.1, z=1.5, speed=65.0),
            confidence=0.9, player_detections=[], timestamp=0.066, frame_number=2,
        ),
    ]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0, "cam2": 1.0})
    curr_pos = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    # Check wall collision
    hit = wall_detector.check(curr_pos, prev_pos)
    assert hit is not None
    assert hit["wall_id"] == "back_near_glass"
    assert hit["surface_type"] == "glass"


def test_no_wall_hit_during_normal_rally(court):
    """Ball stays in middle of court, no wall hit."""
    fusion = WorldFusion(court)
    wall_detector = WallCollisionDetector(court)

    obs1 = [CameraObservation(
        camera_id="cam1", ball_pixel=(500, 300),
        ball_bbox=[490, 290, 510, 310],
        ball_court=BallPosition(x=5.0, y=8.0, z=1.0, speed=50.0),
        confidence=0.9, player_detections=[], timestamp=0.033, frame_number=1,
    )]
    ws1 = fusion.fuse(obs1, {"cam1": 1.0})

    obs2 = [CameraObservation(
        camera_id="cam1", ball_pixel=(500, 350),
        ball_bbox=[490, 340, 510, 360],
        ball_court=BallPosition(x=5.0, y=12.0, z=0.8, speed=45.0),
        confidence=0.9, player_detections=[], timestamp=0.066, frame_number=2,
    )]
    ws2 = fusion.fuse(obs2, {"cam1": 1.0})

    prev = {"x": ws1.ball.x, "y": ws1.ball.y, "z": ws1.ball.z, "speed": ws1.ball.speed}
    curr = {"x": ws2.ball.x, "y": ws2.ball.y, "z": ws2.ball.z, "speed": ws2.ball.speed}

    hit = wall_detector.check(curr, prev)
    assert hit is None
```

- [ ] **Step 2: Run integration tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_multi_camera_integration.py -v`
Expected: All PASS

- [ ] **Step 3: Run full test suite**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/tests/test_multi_camera_integration.py
git commit -m "test: end-to-end multi-camera fusion with wall hit detection"
```

---

## Task 14: Update Results Storage for Wall Hits and World Trajectory

**Files:**
- Modify: `backend/main.py` (results saving in `run_analysis` and `start_analysis`)
- Modify: `backend/tests/test_api.py`

- [ ] **Step 1: Update results.json serialization**

In `backend/main.py`, in both `run()` (line ~388) and `run_analysis()` (line ~558) functions that write `results.json`, add:

```python
# Add wall_hits from events
wall_hits = [
    {
        "event_type": e.event_type.value,
        "frame_number": e.frame_number,
        "timestamp": e.timestamp,
        "metadata": e.metadata,
    }
    for e in analyzer.all_events
    if e.event_type == EventType.WALL_HIT
]
```

Add `"wall_hits": wall_hits` and `"world_trajectory": analyzer.ball_tracker.trajectory` to the JSON dump.

- [ ] **Step 2: Add cameras list to match config persistence**

In `_save_match()` and `_load_match()`, ensure the `cameras` list is preserved. When creating a match via `POST /match/setup`, initialize `"cameras": []`.

- [ ] **Step 3: Run API tests**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_api.py -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
cd /Users/jonathan/Documents/Github/padel_analyzer
git add backend/main.py backend/tests/test_api.py
git commit -m "feat: persist wall_hits and cameras in results and match config"
```

---

## Follow-Up Tasks (deferred from spec, not blocking core functionality)

These spec requirements build on the core foundation above. Each can be implemented independently after Tasks 1-14 are complete:

### Follow-Up A: Stereo Triangulation
When 2+ cameras see the ball with >15° angular separation, use raw pixel positions + camera models to triangulate true 3D position instead of weighted averaging. Add to `WorldFusion._fuse_ball()`. Requires computing angular separation from camera positions and implementing `cv2.triangulatePoints()`.

### Follow-Up B: Frame Synchronization
Live mode: reject observations with timestamp drift >30ms from fusion cycle. Post-match: add frame offset per camera (user-provided or auto-sync via audio cross-correlation). Add sync logic to `WorldFusion.fuse()`.

### Follow-Up C: Adaptive Quality Scaling
If WorldFusion FPS drops below 30, signal CameraNodes to increase frame skip or switch to FastBallDetector. Requires FPS measurement in WorldFusion and a feedback channel to CameraNodes.

### Follow-Up D: Camera Health Events
When `CameraNode.update_health()` transitions to DEGRADED or DISCONNECTED, emit a `MatchEvent` (new `EventType.CAMERA_HEALTH`) so the frontend can display warnings.

### Follow-Up E: Observation Queue for Threading
Replace synchronous `WorldFusion.fuse()` calls with a bounded `queue.Queue` (max size = 2 × num_cameras) between CameraNode threads and WorldFusion thread. Drop oldest on overflow.
