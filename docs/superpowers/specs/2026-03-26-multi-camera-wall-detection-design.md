# Multi-Camera Fusion & Wall Hit Detection

## Overview

Extend the padel analyzer from single-camera to N-camera support with a unified 3D world model. Wall/fence hit detection becomes a 3D geometry operation against court wall planes. All coordinates normalized to standard padel court dimensions (10m x 20m). Designed for both live (RTSP) and post-match (pre-synced video) modes.

## 1. Standard Court Model (`PadelCourtModel`)

Canonical 3D geometry of a padel court in meters. Single source of truth for all coordinate references.

### Court Surface (Z=0)
- Dimensions: 10m wide (X: 0-10), 20m long (Y: 0-20)
- Net at Y=10, service lines at Y=6.95 and Y=13.05
- Center service line at X=5

### Wall Geometry (3D planes)
- **Back walls** (Y=0 and Y=20): 4m high glass (Z: 0-4), optional 1m mesh above (Z: 4-5)
- **Side walls**: 3m glass extending 4m from each back wall (Y: 0-4 and Y: 16-20), then fence for remaining length
- Each wall segment is a bounded 3D plane with `wall_id`, `surface_type` (glass, mesh, fence, open), plane definition (point + normal), and min/max corners

### Optional Overrides
- User can provide wall/cage reference points during calibration to adjust wall positions/heights from standard dimensions
- Stored per-match (describes the physical court, not a camera)
- If not provided, standard padel dimensions are used

### Replaces Scattered Constants
The existing `x_min/x_max/y_min/y_max` in `EventDetectorConfig` and hardcoded dimensions in `CourtCalibration` are consolidated into `PadelCourtModel`. All components reference the model instead of local constants. `EventDetectorConfig.enclosure_bounds` is removed; code that previously read bounds from config now queries `PadelCourtModel.get_bounds()`.

## 2. Multi-Camera Architecture

### CameraNode (one per camera)

Each camera feed is wrapped in a `CameraNode` that runs independently.

**Owns:**
- `CameraModel` (3D calibration via solvePnP, as today)
- Ball detector (configurable: YOLO, TrackNet, or FastBallDetector)
- Player detector (YOLO, shared `UnifiedYoloDetector`)
- `CourtDetector` for continuous auto-recalibration

**Runs:**
- Own thread (live mode) or own video reader (post-match)
- Continuously recalibrates via court keypoint auto-detection against `PadelCourtModel`
- Tracks its own quality score: calibration reprojection error, detection rate, viewing angle

**Outputs per frame:**
- `CameraObservation`: camera_id, raw pixel ball detection (bbox + center), camera's own court-coordinate projection of the ball, confidence, player detections (both pixel and court coords), timestamp

Raw pixel detections are preserved in the observation so that `WorldFusion` can perform stereo triangulation from pixel positions + camera models when 2+ cameras see the ball. The pre-projected court coordinates are used for the weighted-average fallback path.

### WorldFusion (single instance)

Consumes observations from all CameraNodes and produces a unified world state.

**Ball fusion:**
- Weighted average of all camera court-coordinate projections
- Weight = `confidence x (1 / reprojection_error) x viewing_angle_factor`
- 2+ cameras seeing ball simultaneously: triangulation from raw pixel positions + camera models for true 3D position (requires minimum 15-degree angular separation between camera rays; below that threshold, falls back to weighted monocular averaging)
- 1 camera: monocular height estimation (current behavior)
- 0 cameras: Kalman prediction from last known state

**Player fusion:**
- Merge player detections across cameras by proximity in court coordinates
- Deduplicate (same player seen by multiple cameras)
- Keep best bounding box per player

**Output:**
- `WorldState` per frame: fused ball position (x, y, z), velocity vector, fused player positions, contributing camera IDs, timestamp

**Observation queue:**
- CameraNodes push observations to a bounded `queue.Queue` (max size = 2 x num_cameras)
- If queue is full, oldest observation is dropped (non-blocking put with discard)
- WorldFusion consumes all available observations per fusion cycle

### Camera Health & Failure

Each `CameraNode` reports a heartbeat (last frame timestamp) to `WorldFusion`. Failure handling:

- **No frames for 2 seconds**: camera marked `degraded`, quality weight set to 0 (excluded from fusion), user notified via event
- **No frames for 10 seconds**: camera marked `disconnected`, removed from active camera list
- **Recovery**: if frames resume, camera re-enters as `degraded` until calibration quality is verified (reprojection error < threshold), then returns to `active`
- **Analysis continues** with remaining cameras — never pauses unless all cameras are down

### Calibration Quality Thresholds

- Reprojection error **< 5px**: `good` — full weight in fusion
- Reprojection error **5-15px**: `acceptable` — reduced weight (weight *= 0.5)
- Reprojection error **> 15px**: `poor` — camera excluded from fusion, warning emitted, triggers recalibration attempt
- These thresholds apply both at setup time (cross-camera validation) and during dynamic recalibration

### Synchronization
- **Live (RTSP):** sync by timestamp, configurable max drift tolerance (default 30ms)
- **Post-match:** sync by frame offset per camera. User provides initial frame offsets (e.g., camera B starts 45 frames after camera A), or the system auto-syncs via audio cross-correlation if audio tracks are available. Once aligned, sync by frame number.
- **Mixed FPS:** interpolate to highest FPS camera's timeline

### Single Camera Compatibility
A single-camera match is WorldFusion with one CameraNode. Behaves identically to current system.

## 3. Wall/Fence Hit Detection (`WallCollisionDetector`)

New detector in `logic/detectors/`. Operates on fused 3D ball trajectory from WorldFusion.

### Detection Method
1. Each frame, take the ball trajectory segment between frame N-1 and frame N (two 3D points)
2. Ray-plane intersection test against all wall segment planes from `PadelCourtModel`
3. If intersection point falls within the wall segment's bounds (correct height and length): candidate hit
4. Confirm with velocity direction change: ball's velocity vector should reflect off the wall normal (dot product sign flip on the wall's perpendicular axis)

### Event Data (`EventType.WALL_HIT`)
Each wall hit event captures:
- `wall_id`: which wall segment (left_glass, right_glass, back_near, back_far, side_fence, etc.)
- `surface_type`: glass, mesh, or fence
- `impact_point_3d`: precise (x, y, z) on the wall surface
- `speed_at_impact`: ball speed in km/h
- `incoming_angle`: angle of incidence relative to wall normal

### Pipeline Integration

`WallCollisionDetector` is added as a sub-detector inside `EventDetector`, alongside the existing `BounceDetector`, `ServeDetector`, `LastHitterDetector`, and `PointEndDetector`.

**Call chain:**
1. `WorldFusion` produces a `WorldState` per frame
2. `VideoAnalyzer` converts `WorldState` to the existing dict format that `EventDetector.process()` expects: `ball_pos = {"x": ws.ball.x, "y": ws.ball.y, "z": ws.ball.z, "speed": ws.ball.speed}`, `player_positions = [{"x": p.x, "y": p.y, ...} for p in ws.players]`
3. `EventDetector.process()` calls each sub-detector as today, plus the new `WallCollisionDetector.check(ball_pos, ball_velocity, court_model)`
4. Wall hit events are appended to the event list like any other event

This keeps `EventDetector.process()` signature unchanged — the adaptation happens in `VideoAnalyzer`. The `WorldState` dataclass is used internally by the pipeline but does not leak into the logic layer's interface.

### Replaces `PointEndDetector` Wall Logic

The existing wall-proximity heuristic in `PointEndDetector` (lines 35-44: 2D bounds check with 0.3m margin) is **removed**. Instead:
- `PointEndDetector` listens for `WALL_HIT` events from `WallCollisionDetector`
- If a `WALL_HIT` occurs before the first bounce on the receiving side, `PointEndDetector` triggers `PointReason.WALL_BEFORE_BOUNCE`
- This is more accurate (uses 3D collision) and eliminates the hardcoded margin

**2D fallback**: If no 3D calibration is available (no `CameraModel`), `WallCollisionDetector` falls back to a 2D proximity check against `PadelCourtModel` court bounds — equivalent to the old behavior but sourced from the court model instead of `EventDetectorConfig`.

### Game Logic Integration
- `PointEndDetector`: consumes wall hit events for rule enforcement (wall before bounce = point lost)
- `EventDetector`: emits `WALL_HIT` for every wall contact, not just point-ending ones (enables analytics)
- `MatchStateMachine`: no changes needed — wall hits during rally are valid play and don't trigger state transitions. The state machine only transitions on point-ending events, which are already handled by `PointEndDetector`.

### Graceful Degradation
- **2+ cameras with good 3D**: precise ray-plane intersection
- **1 camera with monocular Z**: intersection still works, lower Z accuracy, increased wall proximity threshold
- **No wall reference points**: standard padel dimensions from `PadelCourtModel`
- **No 3D calibration at all**: falls back to 2D proximity check against court model bounds

## 4. Real-Time Performance

### Per-Camera Processing
- Each `CameraNode` runs on its own thread — cameras process in parallel
- Detector choice per mode:
  - **Live**: `FastBallDetector` (2-5ms) default, frame skipping (every 2nd-3rd frame)
  - **Post-match**: `TrackNet` or `YOLO` (50-100ms) for maximum accuracy
  - User-configurable override per camera

### WorldFusion Budget
- Fusion: weighted averaging + optional triangulation = sub-millisecond
- Wall collision: ~6-10 plane intersection tests per frame = negligible
- Runs on its own thread, consumes from shared observation queue

### Scaling
- Cameras are embarrassingly parallel (independent threads)
- CPU-bound: ~1 core per camera for live with FastBallDetector
- GPU-bound: YOLO/TrackNet share GPU — for live multi-camera, limit neural net detectors to 1-2 cameras, FastBallDetector on rest
- Practical limit: 4 cameras live on decent machine (4+ cores, 1 GPU)

### Adaptive Quality
- If fusion FPS drops below 30 FPS target, WorldFusion signals CameraNodes to increase frame skip or switch to faster detector
- Priority: camera with best current view of ball gets full processing, others skip more aggressively

### Post-Match Optimization
- No real-time constraint — process each frame fully across all cameras
- Batch GPU inference: stack frames from multiple cameras into single YOLO/TrackNet call
- Per-camera-per-frame parallelism without time pressure

## 5. Calibration & Setup Flow

### Per-Camera Calibration
- Each camera calibrates independently against `PadelCourtModel`
- All existing flows work: manual 4-corner, 12-keypoint, auto-detect
- New optional step: mark wall/cage reference points (top corners of back glass, fence post tops) on any one camera's view
- Wall reference points stored on the match, not per camera

### `CourtCalibration` Migration

The existing `CourtCalibration` class (2D homography) is **retained but wrapped**. `CameraModel` becomes the primary calibration interface for all components:

- `CameraModel` gains adapter methods matching `CourtCalibration`'s interface: `pixel_to_court(px, py)` delegates to `project_to_ground(px, py)`, `court_to_pixel(cx, cy)` delegates to `court_to_pixel(cx, cy, cz=0)`
- `BallTracker` and `PlayerTracker` are updated to accept `CameraModel` instead of `CourtCalibration`
- `CourtCalibration` remains available as the internal 2D fallback inside `CameraModel` when solvePnP calibration is unavailable
- `CameraNode` always exposes a `CameraModel`; the trackers never interact with `CourtCalibration` directly

### Cross-Camera Validation
- After 2+ cameras calibrated, WorldFusion projects shared court keypoints through each camera's model
- Reports reprojection error per camera — helps user spot bad calibration
- Cameras exceeding the 15px reprojection error threshold are flagged with a warning
- No manual cross-camera calibration step — shared court coordinate system is the implicit link

### Dynamic Recalibration
- Each CameraNode continuously auto-detects court keypoints via `CourtDetector`
- If keypoints shift (camera bumped), recalibrates its own `CameraModel`
- WorldFusion adjusts that camera's quality weight based on calibration freshness and reprojection error

### Setup UI Flow
1. Create match (as today)
2. Add cameras — each gets a name/label (e.g., "Center High", "Corner CCTV")
3. For each camera: upload frame or connect RTSP, run calibration
4. Optionally: mark wall reference points on any one camera's view
5. System validates cross-camera consistency, shows quality scores per camera
6. For post-match with multiple videos: set frame offsets to align start times (or use auto-sync via audio cross-correlation)
7. Start analysis

### Backward Compatibility
Single-camera match skips steps 2 and cross-camera validation. Existing matches keep working unchanged.

## 6. Data Model Changes

### Match Config (`config.json`)
New fields:
- `cameras`: list of `{camera_id, label, camera_model, source_type, source_path}`
- `court_model_overrides`: optional wall reference points overriding `PadelCourtModel` defaults

Existing single-camera fields (`calibration`, `camera_model`) become the first entry in `cameras` for backward compatibility.

### New Types (`models/types.py`)
- `WallSegment`: wall_id, surface_type, plane (point + normal), bounds (min/max corners)
- `CameraObservation`: camera_id, raw pixel detection (bbox, center), court-coordinate projection, player detections (pixel + court), confidence, timestamp
- `WorldState`: fused ball (x, y, z), velocity, player positions, contributing_cameras, timestamp
- `WallHitEvent` metadata: wall_id, surface_type, impact_point_3d, speed_at_impact, incoming_angle

### API Changes
New endpoints:
- `POST /match/{id}/cameras` — add a camera to a match
- `POST /match/{id}/cameras/{cam_id}/calibrate` — calibrate one camera
- `POST /match/{id}/court-model` — set wall reference point overrides
- `GET /match/{id}/world-state` — fused world state per frame

Existing endpoints (`/trajectory`, `/positions`, `/events`) return fused data from WorldFusion — no breaking change for frontend consumers.

### Results Storage
`results.json` gains:
- `wall_hits`: array of wall hit events
- `world_trajectory`: 3D fused positions per frame

Existing `trajectory` and `player_positions` fields kept, populated from fused data.

## 7. Live Mode Architecture

The existing `LiveManager` evolves to orchestrate multiple `CameraNode` instances:

- `LiveManager` creates one `CameraNode` per RTSP feed
- Each `CameraNode` runs its own capture + detection thread (replacing the single `_run_loop`)
- `LiveManager` owns the `WorldFusion` instance, which consumes from all CameraNodes
- `LiveManager` continues to own the WebSocket streaming, replay buffer, and event queue — these now source data from `WorldFusion.get_latest_state()` instead of a single `VideoAnalyzer`
- The single `VideoAnalyzer` is no longer used directly in live mode; its logic is split between `CameraNode` (per-camera detection) and `WorldFusion` (fusion + event detection)

For post-match mode, `VideoAnalyzer` is similarly refactored: it creates N `CameraNode` instances (one per video file), feeds them through `WorldFusion`, and runs `EventDetector` on the fused output.

## 8. New File Structure

```
backend/src/
  cv/
    camera_node.py          # CameraNode: per-camera processing wrapper
  logic/
    detectors/
      wall_collision.py     # WallCollisionDetector
  models/
    court_model.py          # PadelCourtModel: 3D court + wall geometry
    types.py                # Extended with WallSegment, CameraObservation, WorldState
  pipeline/
    world_fusion.py         # WorldFusion: multi-camera observation merging
    video_analyzer.py       # Refactored to use WorldFusion + CameraNodes
    live_manager.py         # Refactored for multi-RTSP via CameraNodes
```
