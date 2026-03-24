# Phase 2: Full CV Pipeline — Offline Analysis + Live Mode

## Overview

Build the complete computer vision pipeline that turns raw video (file or camera) into automatically scored padel matches with real-time event detection, trajectory tracking, and live streaming.

**Scope:** Offline video analysis + live camera mode in one phase.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Ball detection | YOLO cls 32 now, TrackNetV2-ready interface | Get working pipeline fast; Kalman compensates for misses |
| Player tracking | Keep IoU matching | 4 players in fixed court halves, minimal occlusion |
| Test footage | Real padel matches from YouTube | Actual detection challenges |
| Frontend | Minimal functional UI, backend-first | Focus on the hard CV problems |
| Hardware | MPS primary, CUDA secondary, CPU fallback | Apple Silicon dev machine, abstract for portability |

---

## Section 1: Detection Layer

Abstract detector interfaces with YOLO concrete implementations. Single YOLOv8n inference pass per frame, results split by class.

### Interfaces

**BallDetector** (abstract):
- `detect(frame: np.ndarray) -> Optional[List[float]]` — returns `[x1, y1, x2, y2]` bbox or None
- `device` property — current compute device string
- `warm_up()` — run dummy inference to avoid first-frame latency

**PlayerDetector** (abstract):
- `detect(frame: np.ndarray) -> np.ndarray` — returns N×6 array `[x1, y1, x2, y2, conf, cls]`
- `device` property
- `warm_up()`

### Concrete Implementations

**YoloBallDetector**:
- Model: `yolov8n.pt`
- Class filter: cls 32 (sports ball)
- Confidence threshold: 0.3 (low — Kalman filter compensates for missed frames)
- Returns highest-confidence detection if multiple found

**YoloPlayerDetector**:
- Model: `yolov8n.pt` (same model instance, shared with ball detector)
- Class filter: cls 0 (person)
- Confidence threshold: 0.5
- Max detections: 4 (padel court constraint)
- Returns top-4 by confidence

**Shared model optimization**: Both detectors share a single YOLO model instance. One `model(frame)` call per frame, results filtered by class ID in each detector. Avoids running inference twice.

### Device Auto-Detection

`get_device()` utility function:
1. Check `torch.backends.mps.is_available()` → return "mps" (Apple Silicon)
2. Check `torch.cuda.is_available()` → return "cuda"
3. Fallback → return "cpu"

Set once at model initialization, used for all inference.

**Warm-up**: First YOLO inference is slow (model load + JIT compilation). `warm_up()` runs a dummy 640×640 frame through the model at init time so real frames don't hit a latency spike.

### Future: TrackNetV2

The `BallDetector` interface accepts a single frame. TrackNetV2 needs 3 consecutive frames. The interface will be extended with an optional `detect_temporal(frames: List[np.ndarray])` method when TrackNetV2 is integrated. The `VideoAnalyzer` pipeline already maintains a frame buffer that can supply this.

### File Structure

```
src/cv/detectors/
├── __init__.py
├── base.py          # ABC: BallDetector, PlayerDetector
├── yolo.py          # YoloBallDetector, YoloPlayerDetector
└── device.py        # get_device() utility
```

Existing files unchanged: `ball_tracker.py`, `player_tracker.py`, `court_calibration.py`.

---

## Section 2: Event Detection & State Machine

The layer between trackers and the scoring engine. Consumes ball + player positions per frame, emits `MatchEvent` objects, drives the scoring engine.

### Match State Machine

States and transitions:

```
IDLE
  → ball detected in service zone → SERVING_1ST

SERVING_1ST
  → fault (net/out/wrong service box) → SERVING_2ND
  → let (net hit + valid landing) → SERVING_1ST (replay)
  → valid serve (bounces in correct diagonal service box) → RALLY

SERVING_2ND
  → fault → POINT_ENDED (reason: DOUBLE_FAULT, receiver wins)
  → let → SERVING_2ND (replay)
  → valid serve → RALLY

RALLY
  → double bounce on one side → POINT_ENDED
  → ball out of bounds → POINT_ENDED
  → ball hits net and stops → POINT_ENDED
  → wall before bounce (illegal) → POINT_ENDED
  → ball lost for too long (tracker lost) → POINT_ENDED
  → winner (unreturned shot) → POINT_ENDED

POINT_ENDED
  → determine winner → scoring_engine.add_point(team, reason)
  → SCORE_UPDATE

SCORE_UPDATE
  → emit events → reset detectors → IDLE
```

### Sub-Detectors

**BounceDetector**:
- Triggers when: ball Z estimate drops to ~0 AND ground-plane speed dips AND ball was descending (pixel-Y increasing)
- Records: bounce position in court coords, court side (near/far), timestamp, frame number
- Maintains bounce count per side for double-bounce detection

**ServeDetector**:
- Triggers when: ball first appears near current server's position, moves toward opponent side
- Validates: first bounce lands in correct diagonal service box (uses `CourtCalibration.is_in_service_box()`)
- Detects let: ball clips net (Z dip at y≈10m) but still lands in valid service box
- Knows current server from `ScoringEngine.current_server`

**LastHitterDetector**:
- Tracks ball velocity direction frame-to-frame
- When velocity direction reverses significantly (ball changes heading >90°), finds closest assigned player via `PlayerTracker.find_closest_player(ball_x, ball_y)`
- That player's team is the "last hitter" — used to determine who loses on errors

**PointEndDetector**:
- Watches for 5 end conditions:
  1. Double bounce: 2 bounces on same side without ball crossing net (y=10m line)
  2. Ball out of bounds: `CourtCalibration.is_in_bounds()` returns False after a bounce
  3. Ball stopped: speed ≈ 0 for 15+ consecutive frames
  4. Ball lost: `BallTracker.is_lost` is True (60+ frames without detection)
  5. Winner: ball bounces on one side, no return within N frames (configurable timeout)
- Emits `PointReason` enum matching the existing `types.py` values

### Point Winner Logic

5 rules for determining which team wins:

| End Reason | Winner |
|-----------|--------|
| DOUBLE_FAULT | Receiving team |
| DOUBLE_BOUNCE | Team on the OTHER side from where bounces occurred |
| OUT | Team that did NOT hit last |
| NET (ball stops) | Team that did NOT hit last |
| WINNER | Team that hit last (unreturnable shot) |

### EventDetector (Orchestrator)

Coordinates all sub-detectors per frame:

```python
def process(self, ball_pos, player_positions, frame_no):
    bounce = self.bounce_detector.check(ball_pos)
    hit = self.last_hitter_detector.check(ball_pos, player_positions)
    serve = self.serve_detector.check(ball_pos, bounce)
    point_end = self.point_end_detector.check(bounce, ball_pos)

    events = [e for e in [bounce, hit, serve, point_end] if e]
    self.state_machine.handle(events)
    return events
```

### File Structure

```
src/logic/
├── scoring_engine.py       # existing — unchanged
├── event_detector.py       # EventDetector: orchestrates sub-detectors
├── match_state_machine.py  # MatchStateMachine: state transitions
└── detectors/
    ├── __init__.py
    ├── bounce.py            # BounceDetector
    ├── serve.py             # ServeDetector
    ├── last_hitter.py       # LastHitterDetector
    └── point_end.py         # PointEndDetector
```

---

## Section 3: Pipeline Orchestrator & API

### VideoAnalyzer

Central class that wires the full pipeline. Accepts either a video file path (offline) or camera device ID (live).

```python
class VideoAnalyzer:
    def __init__(self, match_id, calibration, config):
        self.ball_detector = YoloBallDetector(model)
        self.player_detector = YoloPlayerDetector(model)  # shared model
        self.ball_tracker = BallTracker(calibration)
        self.player_tracker = PlayerTracker(calibration)
        self.event_detector = EventDetector(calibration, config)
        self.scoring_engine = PadelScoringEngine(...)

    def process_frame(self, frame, frame_no):
        ball_bbox = self.ball_detector.detect(frame)
        player_detections = self.player_detector.detect(frame)
        ball_pos = self.ball_tracker.update(ball_bbox, frame_no)
        player_pos = self.player_tracker.update(player_detections, frame_no)
        events = self.event_detector.process(ball_pos, player_pos, frame_no)
        return FrameResult(ball_pos, player_pos, events, self.scoring_engine.get_score_display())
```

**Offline mode** (`analyze_video(path)`):
- Opens video with `cv2.VideoCapture`
- Processes all frames sequentially
- Reports progress via callback (percentage, fps, ETA)
- Saves results to `data/matches/{id}/results.json`
- Returns job ID for status polling

**Live mode** (`start_live(device_id)`):
- Opens camera with `cv2.VideoCapture(device_id)` or RTSP URL
- Processes frames in real-time in a background thread
- Pushes events/score/frames to WebSocket clients
- Maintains 30-second replay buffer
- Optional recording to disk via separate writer thread

### API Endpoints

**Existing (unchanged):**
- `GET /` — health check
- `POST /match/setup` — create match config
- `GET /match/{id}` — get match config
- `POST /match/{id}/calibrate` — set court corners

**New — Offline Analysis:**
- `POST /analyze/upload` — upload video file, returns `{job_id}`
- `POST /analyze/start/{job_id}` — start background processing
- `GET /analyze/status/{job_id}` — progress: `{state, percent, fps, eta}`

**New — Results:**
- `GET /match/{id}/score` — current score display
- `GET /match/{id}/events` — all match events
- `GET /match/{id}/trajectory` — ball trajectory data
- `GET /match/{id}/stats` — player stats, heatmaps
- `POST /match/{id}/correct-score` — manual score override

**New — Live Mode:**
- `POST /live/start` — start camera feed `{device_id | rtsp_url, match_id}`
- `POST /live/stop` — stop camera, save recording
- `WS /live/stream` — bidirectional WebSocket
- `GET /live/replay` — last 30s replay buffer

### WebSocket Protocol (`/live/stream`)

**Server → Client:**
- `{type: "frame", jpeg: <base64>}` — every frame (~30fps)
- `{type: "score", data: <scoreDisplay>}` — on score change
- `{type: "event", data: <matchEvent>}` — on event detection
- `{type: "status", fps: 28, latency: 19}` — periodic health

**Client → Server:**
- `{type: "correct", team: 1}` — manual score correction
- `{type: "reassign", track_id: 3, player_id: "P2"}` — reassign player
- `{type: "replay"}` — request 30s replay

### Replay Buffer

Ring buffer of 900 JPEG frames (30s × 30fps). Quality 70 (~30KB/frame = ~27MB memory). Circular — overwrites oldest frame. On replay request, returns buffer contents as ordered frame list.

### Full Match Recording (optional)

Separate thread writes frames to disk via `cv2.VideoWriter` with H.264 codec. Toggled at `POST /live/start` with `{record: true}`. Saves to `data/matches/{id}/recording.mp4`.

### File Structure

```
src/pipeline/
├── __init__.py
├── video_analyzer.py    # VideoAnalyzer: orchestrates full pipeline
├── live_manager.py      # LiveManager: camera feed + WebSocket push
└── replay_buffer.py     # ReplayBuffer: 30s JPEG ring buffer
```

---

## Section 4: Minimal Frontend

Backend-first approach — frontend gets just enough to demonstrate the system works.

### Pages

1. **Match Setup** — form: match name, player names, teams, format, golden point, first server. POST to `/match/setup`.
2. **Calibration** — upload video or connect camera, click 4 court corners on the frame, POST to `/match/{id}/calibrate`. Show digital twin preview.
3. **Offline Analysis** — upload video, start analysis, show progress bar, display results (events list, score, trajectory).
4. **Live View** — connect to WebSocket, show video feed, scoreboard overlay, event log, replay button, manual score correction.

### Tech

Keep existing React + Three.js setup. Add:
- React Router for page navigation
- WebSocket client hook for live mode
- Simple scoreboard component

No charts, heatmaps, 3D replay, or PDF export in this phase. Those are Phase 3.5 polish.

---

## Section 5: Testing Strategy

### Unit Tests (per module)

- **Detectors**: Mock YOLO model, verify class filtering, confidence thresholds, device selection
- **BounceDetector**: Feed synthetic ball positions with known Z/speed patterns, verify bounce detection
- **ServeDetector**: Feed ball trajectory from service zone to service box, verify serve detection and fault cases
- **LastHitterDetector**: Feed velocity changes with known player positions, verify correct player identified
- **PointEndDetector**: Feed sequences for each end condition (double bounce, out, net, lost, winner)
- **MatchStateMachine**: Feed event sequences, verify correct state transitions
- **EventDetector**: Integration of sub-detectors with state machine
- **VideoAnalyzer**: Mock detectors, verify frame processing pipeline
- **ReplayBuffer**: Verify ring buffer behavior, capacity, ordering
- **API endpoints**: Test new endpoints with httpx (same pattern as existing tests)

### Integration Tests

- **Full pipeline with synthetic data**: Construct frame sequences with known ball/player positions, run through full pipeline, verify correct score output
- **Full pipeline with real video**: Process a short YouTube clip, verify no crashes, reasonable event detection

### Test with Real Footage

Download a short padel match clip from YouTube for development testing. Use `yt-dlp` to grab a 2-3 minute segment. Store in `data/test_footage/` (gitignored via `*.mp4` rule).

---

## Dependencies

### New Python packages
- None required — `ultralytics`, `torch`, `opencv-python`, `filterpy` already in `requirements.txt`

### New Frontend packages
- `react-router-dom` — page routing

### Optional (for dev convenience)
- `yt-dlp` — download test footage (dev tool, not a project dependency)
