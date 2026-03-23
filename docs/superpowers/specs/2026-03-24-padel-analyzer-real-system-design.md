# Padel Analyzer — Full System Design Spec

**Date:** 2026-03-24
**Status:** Approved
**Scope:** Transform prototype into real CV-based padel match analyzer

## 1. Goals

The system provides full padel match analysis:
- (A) Automatic score tracking with padel rules (golden point, deuce, sets)
- (B) Player analytics — heatmaps, positioning, movement, per-player stats
- (C) Ball trajectory reconstruction — 3D path, speed estimation, trail visualization
- (D) Full match analysis — highlights, replays, statistics, export

## 2. Input Sources

Two input modes, same pipeline:
- **File mode:** Upload mp4/mov video, process faster than real-time
- **Live mode:** USB camera (`cv2.VideoCapture(0)`) or RTSP/IP stream (`cv2.VideoCapture("rtsp://...")`)

Both feed into identical detection → tracking → events → scoring pipeline. Live mode drops frames if pipeline can't keep up with frame rate.

## 3. Architecture

```
Input (File or Live Stream)
    │
    ▼
┌─────────────────────────────────────┐
│  1. COURT CALIBRATION               │
│  User clicks 4 court corners        │
│  → homography matrix (px → meters)  │
│  Optional: net posts + service lines │
│  Optional: auto-detect via Hough    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  2. DETECTION (per frame)           │
│  YOLOv8n   → player bounding boxes  │
│  Ball: YOLO cls 32 (Phase 1)       │
│        TrackNetV2 (Phase 2+)        │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  3. TRACKING                        │
│  ByteTrack    → persistent player   │
│                  IDs (P1-P4)        │
│  Kalman Filter → ball smoothing     │
│                  through occlusion  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  4. EVENT DETECTION                 │
│  State machine:                     │
│  IDLE → SERVING → RALLY →          │
│  POINT_ENDED → SCORE_UPDATE → IDLE │
│                                     │
│  Detects: bounce, serve, fault,     │
│  net hit, wall/fence hit, out,      │
│  point end, rally count             │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  5. SCORING ENGINE (existing)       │
│  PadelScoringEngine                 │
│  Events → points → games → sets    │
│  Supports golden point + deuce/AD   │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  6. OUTPUT                          │
│  Annotated video overlays           │
│  Player heatmaps + stats            │
│  Ball trajectory maps               │
│  3D court replay (Three.js)         │
│  Auto-generated highlights          │
│  Export: PDF, JSON, highlight reel  │
└─────────────────────────────────────┘
```

Backend processes video as a job, streams progress to frontend via WebSocket, saves results as JSON.

## 4. Court Calibration

**Replaces** the current hardcoded `CameraCalibration` class that requires knowing exact camera position/height/tilt.

**Coordinate system:** X-axis = court width (0-10m, left to right from camera view). Y-axis = court length (0-20m, near baseline to far baseline). Origin (0,0) = near-left corner. Net line at y=10m. Service boxes: y=6.95m to y=13.05m, divided at x=5m.

**Approach:** Homography from user-marked points.

1. User sees camera preview (live feed or first frame of video)
2. User clicks 4 court corners (baseline-sideline intersections) — minimum requirement
3. System computes homography matrix: pixels → 10x20m court coordinates
4. Optional: click net posts + service lines for 8-point accuracy
5. **Auto-Detect Corners** button: Hough line detection to find court lines automatically, falls back to manual
6. Digital Twin preview (3D court in Three.js) updates live as corners are clicked — shows estimated camera position for verification
7. Calibration saved per camera setup — reusable across sessions
8. For live mode: calibrate once, stays valid while camera is fixed

## 5. Detection & Tracking

### 5.1 Ball Detection — TrackNetV2

- Purpose-built neural network for small fast ball tracking (originally badminton/tennis)
- Input: 3 consecutive frames (640x360 resized)
- Output: heatmap → ball (x, y) per frame + confidence
- Open source PyTorch implementation
- May need fine-tuning on padel footage (ball similar to tennis)
- Kalman filter on top for smoothing through occlusion

### 5.2 Player Detection — YOLOv8n + ByteTrack

- YOLOv8n (nano variant) for speed — detecting `person` class only (cls 0)
- ByteTrack for multi-object tracking — maintains player IDs across frames
- Player assignment flow:
  1. User enters player names/teams in Match Setup
  2. After calibration, system runs detection on first frame
  3. User clicks each detected bounding box to assign P1/P2/P3/P4
  4. ByteTrack maintains those IDs throughout match
- If ID lost (player leaves frame, heavy occlusion), system flags for user correction

### 5.3 Ball Height (Z) Estimation

Single camera cannot directly measure height. Two complementary methods:

**Method A — Ball size regression:** A padel ball has a known diameter (~6.5cm). Compare detected ball bounding box size to expected size at ground level (known via homography). Larger-than-expected = closer to camera = higher. This gives a rough Z estimate (±0.3m accuracy).

**Method B — Parabolic trajectory fitting:** Ball flight follows a parabola under gravity. Fit a parabolic curve to the last N ball positions in court coordinates. The fitted curve gives Z at each point. Requires ≥5 consecutive detections to fit reliably.

Use Method A for real-time display, Method B for post-hoc trajectory refinement. Both feed into the Kalman filter state.

**Limitation:** Z estimates are approximate. The 2D overhead court view is the primary visualization. The 3D replay uses Z estimates but is presented as "approximate reconstruction" — not ground truth.

### 5.4 Ball Trajectory & Speed

- Kalman filter maintains smoothed trajectory in court coordinates (via homography) with estimated Z
- **Ground-plane speed** = distance between consecutive (x, y) court positions / time between frames. This is the primary speed metric — reliable because homography is accurate on the ground plane.
- Trajectory stored as array: `{x, y, z_estimated, t, speed}` per frame
- Visual "tail" overlay: last 15 positions as fading gradient trail

### 5.5 Frame Processing Budget (live mode, 30fps)

| Stage | Time |
|-------|------|
| TrackNet inference | ~8ms (GPU) |
| YOLOv8n inference | ~6ms (GPU) |
| ByteTrack update | ~1ms (CPU) |
| Kalman filter | ~0.5ms (CPU) |
| Event detection | ~0.5ms (CPU) |
| Overlay rendering | ~3ms (CPU) |
| **Total** | **~19ms** (fits in 33ms budget) |

Without GPU: process every 2nd-3rd frame and interpolate.

## 6. Event Detection

### 6.1 Detected Events

| Event | Detection Method |
|-------|-----------------|
| Bounce | Ball's estimated Z drops to ~0 (via size regression) AND ground-plane speed drops momentarily (ball decelerates on contact). Secondary signal: ball image-Y reaches local maximum (lowest point in frame from behind-baseline view). Position in court coords determines which side. |
| Serve | Ball originates from service zone, moves toward opponent court. First bounce validates in/out. |
| Fault | Serve bounce outside service box (homography gives exact court position). |
| Net hit | Ball trajectory crosses net line (y=10m), speed drops sharply to ~0. |
| Wall/fence hit | Ball's image position enters the wall/fence zone (pixel regions beyond court lines in the frame, NOT homography-projected — homography is only valid on the ground plane). Confirmed by: ball speed direction change + ball remaining in play after. Wall zones defined during calibration. |
| Out | Ball bounces outside court lines without prior wall contact. |
| Point end | Ball stops / leaves frame / double bounce on same side / unrecovered net hit. |
| Rally count | Number of net-line crossings by ball. |

### 6.2 Server Tracking

The system must track who is serving:
- **Initial server:** User selects which team/player serves first in Match Setup
- **Server rotation:** After each game, server alternates between teams (Team A → Team B → Team A). Within each team, players alternate serving games.
- **Tiebreak serving:** Server changes every 2 points, starting with the team that would serve next in rotation.
- Stored in scoring engine state: `current_server: {team_id, player_id}`
- Used for: fault counting (first vs second serve), serve statistics, double fault point attribution

### 6.3 State Machine

```
IDLE → SERVING_1ST ──→ RALLY ──→ POINT_ENDED → SCORE_UPDATE → IDLE
           │               │            ▲
           ▼               │            │
       FAULT_1ST           └────────────┘
           │              (net/out/double bounce
           ▼               ends the rally)
       SERVING_2ND ──→ RALLY
           │
           ▼
       DOUBLE_FAULT → POINT_ENDED
```

- **IDLE:** Waiting for serve detection
- **SERVING_1ST:** First serve in motion, watching bounce location
- **FAULT_1ST:** First serve was out — transition to second serve
- **SERVING_2ND:** Second serve in motion
- **DOUBLE_FAULT:** Second serve also out — point to receiver
- **RALLY:** Ball in play, counting shots, tracking trajectory
- **POINT_ENDED:** Termination event detected — determine winner
- **SCORE_UPDATE:** Feed to PadelScoringEngine, trigger instant replay, rotate server if game won

**Let serve handling:** A let is when the ball clips the net cord but continues into the correct service box. Unlike a net fault (ball stops at the net, speed → ~0), a let shows a partial speed reduction (20-50% drop) as the ball crosses the net line, then continues and bounces in the valid service box. Detection: speed dip at net crossing (but not to zero) + valid bounce within ~1 second. Result: replay the serve (stay in same SERVING state).

**Side changes:** For simplicity in v1, side changes are NOT modeled. Teams stay on their initially assigned sides throughout the match. The system tracks "near side" and "far side" consistently. This is a known simplification — real padel switches sides after odd games. Can be added later by remapping team↔side associations at the appropriate game boundaries.

### 6.4 Last Hitter Detection

Determining which player last hit the ball is required for point attribution.

**Method: proximity + direction change.** When the ball changes direction (velocity vector flips significantly, >90° change), the closest player to the ball at that moment is the hitter. This is a heuristic — not perfect, but reliable for standard play because:
- In padel, players are always close to the ball when hitting
- The ball changes direction sharply on contact
- False positives (ball bouncing near a player) are filtered by requiring the ball to be above ground level (Z estimate > 0.3m) at the direction change

Stored per shot: `{hitter_player_id, timestamp, ball_speed, ball_position}`

### 6.5 Point Winner Logic

After POINT_ENDED:
1. Double bounce on side X → other team wins
2. Ball out (no prior wall contact) → last hitter's team loses
3. Net hit without recovery → last hitter's team loses
4. Ball hits wall/fence before bounce on opponent side → last hitter's team loses (padel rule)
5. Double fault → receiving team wins

### 6.6 Error Recovery (Live Mode)

| Problem | Detection | Recovery |
|---------|-----------|----------|
| Ball lost (>2s) | TrackNet confidence below threshold for 60+ frames | Show "Ball Lost" indicator. Pause event detection. Resume when ball re-detected. |
| Player ID lost | ByteTrack ID disappears | Highlight affected player box in yellow. Show "Re-assign Player" prompt in sidebar. Use last known position to suggest match. |
| Bad calibration | User notices overlay lines don't match court | "Recalibrate" button available at all times in sidebar. Pauses match, returns to calibration screen, preserves score. |
| Score out of sync | User notices wrong score | Manual correction buttons (+/- per team for points, games, sets). Corrections are logged as manual overrides. |

### 6.7 Scoring Engine Updates

The existing `PadelScoringEngine` needs these additions:
- **Match format config:** Constructor accepts `sets_to_win` (1 or 2) instead of hardcoded `== 2`
- **Server tracking:** `current_server` state with `rotate_server()` called on game win
- **Tiebreak mode:** At 6-6 in games, switch to tiebreak scoring (1, 2, 3... instead of 15, 30, 40). Server changes every 2 points. First to 7 with 2-point lead.
- **Let serve:** `register_let()` method — no state change, just logged for stats

## 7. UI/UX Design

### 7.1 Match Setup Screen

Shown before entering either mode.

**Match Info:**
- Match name (text input)
- Format: Best of 3 / Best of 1 / Custom
- Golden Point: Yes / No (Advantage)

**Players:**
- Doubles format (4 players) — standard padel
- Per player: Name, Team (A/B), Position (Left/Right), Color
- First server selection: which team/player serves first
- Team A = near side, Team B = far side

**Camera & Calibration:**
- Input toggle: Upload Video / Live Camera
- Source input: file picker or RTSP URL / device ID
- Camera preview with interactive corner clicking
- Calibration status checklist (4 corners, net posts, service lines)
- Auto-Detect Corners button
- Digital Twin 3D preview (updates as corners are marked)
- Player assignment: click detected bounding boxes to assign to P1-P4

**Saved setups:** Load previous calibration for same camera position.

"Start Match" button enables when calibration complete.

### 7.2 Live Mode

**Layout:**
- **Main area (left 75%):**
  - Top: Live camera feed with overlays (ball trail, player labels, court lines, speed badge, event flash)
  - Middle: Instant replay panel — auto-plays 5-second clip after each point scored. Controls: Prev Point / Replay / Next Point
  - Bottom: Point log — clickable list of all points with timestamp, event description, score
- **Sidebar (right 25%):**
  - Scoreboard: sets, games, current point score
  - Manual score correction: [+] buttons per team
  - Player re-ID button: if ByteTrack loses a player, click to re-assign bounding box to player
  - Live stats: ball speed, rally length
  - Controls: Pause, End Match

**Behaviors:**
- Overlays render in real-time on the feed
- After each point: instant replay auto-plays from 5s before the point
- Recording buffer: rolling 30-second circular buffer of compressed frames (JPEG quality 70, ~100-150KB/frame = ~90-135MB for 30s at 30fps). Used for instant replay. Buffer size is quality-dependent.
- Full match recorded to disk continuously via separate `cv2.VideoWriter` thread (H.264 codec)
- Point log is clickable — jumps to that replay

### 7.3 Playback Mode

**Layout:**
- **Main area (left 75%):**
  - Top: Video playback with overlays + speed control (0.5x, 1x, 2x) + overlay toggles (trajectory, IDs, zones on/off)
  - Middle: Event timeline bar — colored segments (yellow=rally, red=point scored). Click to jump.
  - Bottom left: Auto-generated highlights (longest rallies, fastest shots, winners, errors — clickable clips)
  - Bottom right: Court Replay — 2D overhead view (primary, accurate) with player dots + ball path. Optional 3D mode (Three.js, approximate Z). Synced with video or independent scrub.
- **Sidebar (right 25%):**
  - Match stats: final score, duration, total points, avg rally length
  - Player stats: select player → heatmap + avg speed + winners + errors + court coverage %
  - Export buttons: PDF report, JSON data, highlight reel video

**Analytics computed:**
| Stat | Computation |
|------|-------------|
| Total points | Scoring engine count |
| Average rally length | Net crossings per point |
| Fastest shot | Max ball speed from trajectory |
| Winners | Unreturned in-court shots |
| Unforced errors | Out/net by losing team |
| Court coverage % | Heatmap area / court area per player |
| Serve accuracy % | In-serves / total serves |
| First serve % | First serves in / total service games |

**Highlights auto-generated:**
- Top 5 longest rallies (by shot count)
- Top 5 fastest shots (by ball speed)
- All winners
- All break points

### 7.4 Mode Transitions

- Home screen: choose Live Mode or Playback Mode
- Live → Playback: "End Match" saves recording, opens in Playback mode for analysis
- Playback can also open previously recorded/analyzed matches

## 8. Backend

**FastAPI** with CORS enabled for local dev (`localhost:3000`).

Endpoints:
- `POST /match/setup` — save match config (players, format, calibration)
- `POST /match/calibrate` — receive 4+ corner points, return homography matrix
- `POST /analyze/start` — start processing a video file (returns job ID)
- `GET /analyze/status/{job_id}` — poll processing progress
- `WS /analyze/stream` — WebSocket for live mode. Server→Client messages:
  ```json
  {"type": "frame", "players": [{"id": "P1", "x": 3.2, "y": 8.1}], "ball": {"x": 5, "y": 12, "z": 0.8, "speed": 65.2}, "trail": [[x,y]...]}
  {"type": "event", "event": "BOUNCE", "position": {"x": 3, "y": 5}, "timestamp": 123.4}
  {"type": "score", "score": {"points": "30-15", "games": "3-2", "sets": "1-0"}, "server": "P1"}
  {"type": "status", "fps": 28, "dropped": 2}
  ```
  Client→Server messages:
  ```json
  {"type": "score_correction", "team": 1, "action": "add_point"}
  {"type": "reassign_player", "bbox": [x1,y1,x2,y2], "player_id": "P2"}
  ```
- `GET /match/{id}/events` — get all events for a match
- `GET /match/{id}/stats` — get computed analytics
- `GET /match/{id}/trajectory` — get ball trajectory data
- `GET /match/{id}/heatmap/{player_id}` — get player heatmap data
- `GET /match/{id}/highlights` — get auto-generated highlight clips
- `POST /match/{id}/export/pdf` — generate PDF report

## 9. Data Storage

File-based (SQLite optional later):
```
data/
  matches/
    {match_id}/
      config.json       — match setup, players, calibration
      events.json       — all detected events with timestamps
      trajectory.json   — ball trajectory data (per frame)
      positions.json    — player positions (per frame)
      stats.json        — computed analytics
      recording.mp4     — full match video (live mode)
      highlights/       — extracted highlight clips
```

## 10. Tech Stack

**Backend (Python):**
- FastAPI + uvicorn (API + WebSocket)
- OpenCV (video I/O, image processing, homography)
- PyTorch (model inference)
- Ultralytics YOLOv8n (player detection)
- TrackNetV2 (ball detection)
- ByteTrack (multi-object tracking)
- filterpy (Kalman filter)
- NumPy, SciPy (math)

**Frontend (TypeScript/React):**
- React 18
- Three.js + @react-three/fiber (3D court, digital twin)
- Canvas API (video overlays)
- WebSocket client (live mode streaming)
- Chart library (stats visualization)

## 11. Veo Prompt for Test Footage

To generate test video in Google Veo:

> "A padel match filmed from a fixed elevated camera behind one baseline, approximately 6 meters high, looking down the length of the court. Four players in doubles formation — two in red shirts on the near side, two in blue shirts on the far side. Standard padel court with glass walls visible. The camera is completely static, no movement. Show a full rally: serve from the right service box, ball bounces in the correct service area, players exchange volleys at the net, ball hits the back glass wall and is returned, rally ends with a winning smash. Bright daylight, sharp shadows, the ball is clearly visible throughout. 1080p, 30fps, 15-20 seconds duration."

Generate 3-4 variations with different rally outcomes (winner, net fault, out, wall play) for testing all event types.

## 12. Phased Delivery

### Phase 1: Foundation
- Project restructure, git init, proper config
- Court calibration (interactive 4-corner click + homography)
- YOLOv8n player detection + ByteTrack tracking
- Basic ball detection (YOLO cls 32 as interim before TrackNet)

### Phase 2: Ball Tracking & Events
- TrackNetV2 integration + Kalman filter
- Bounce detection, serve detection, point-end detection
- State machine wired to scoring engine
- Ball trajectory storage + speed computation

### Phase 3: Live Mode
- Camera input (USB/RTSP)
- WebSocket streaming to frontend
- Real-time overlays (ball trail, player labels, score)
- Recording buffer + instant replay
- Match setup screen

### Phase 4: Playback Mode & Analytics
- Video playback with overlays + speed control
- Event timeline bar
- Player heatmaps
- Match statistics computation
- 3D court replay
- Auto-generated highlights

### Phase 5: Export & Polish
- PDF report generation
- JSON export
- Highlight reel stitching
- Saved match library
- UI polish
