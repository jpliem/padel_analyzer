# Phase 3: Frontend Rebuild — Functional UI for Padel Analyzer

## Overview

Replace the prototype frontend with a functional React app that connects to all Phase 2 backend APIs. Dashboard landing, match setup wizard, court calibration with interactive corner clicking, offline video analysis with results, and live camera mode with WebSocket streaming.

**Scope:** Frontend rebuild + one small backend addition (`GET /matches` list endpoint). No TrackNetV2. No advanced analytics (heatmaps, PDF export). Those are separate phases.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Visual style | Clean minimal (Material-ish) | Light background, simple cards, focus on data |
| 3D digital twin | Keep everywhere | Show during calibration, analysis, and live as mini-map |
| Navigation | Dashboard + pages | Landing shows match history, click to view/analyze |
| Panel layout | Resizable panels | `react-resizable-panels` for flexible video/sidebar/court |
| Framework | Evolve CRA prototype | Keep existing setup, add router + panel lib |

## Dependencies

**New npm packages:**
- `react-router-dom` — page routing (5 routes)
- `react-resizable-panels` — draggable panel dividers for analysis/live views

**Existing (keep):**
- `react`, `react-dom` — UI framework
- `three`, `@react-three/fiber`, `@react-three/drei` — 3D court rendering
- `typescript` — type safety

---

## Section 1: Routes & Navigation

### Routes

| Path | Component | Purpose |
|------|-----------|---------|
| `/` | `Dashboard` | Match list with cards, "+ New Match" button |
| `/match/new` | `MatchSetup` | Form: name, players, format, first server |
| `/match/:id/calibrate` | `Calibration` | Click 4 court corners on video frame, 3D preview |
| `/match/:id/analyze` | `OfflineAnalysis` | Upload video, run analysis, view results |
| `/match/:id/live` | `LiveView` | WebSocket camera feed with real-time scoring |

### Navigation Bar

Persistent top bar on all pages:
- Left: App logo/name ("Padel Analyzer"), Dashboard link
- Right: Current match name (if on a match page), connection status indicator (green dot when backend reachable)

### User Flow

```
Dashboard → "+ New Match" → MatchSetup form
  → Submit (POST /match/setup) → redirect to /match/:id/calibrate
  → Click 4 corners → Save (POST /match/:id/calibrate) → choose:
    → "Analyze Video" → /match/:id/analyze
    → "Go Live" → /match/:id/live

Dashboard → click existing match card:
  → If analyzed: go to /match/:id/analyze (shows results)
  → If calibrated only: show "Analyze" / "Go Live" buttons
  → If not calibrated: go to /match/:id/calibrate
```

---

## Section 2: Dashboard Page

### Layout

- Header: "Matches" title + subtitle + "+ New Match" button
- Grid of match cards (responsive, `auto-fill`, `minmax(300px, 1fr)`)
- Empty state: single dashed "+" card when no matches exist

### Match Card

Each card shows:
- Match name + creation time
- Status badge: `Created` (gray), `Calibrated` (yellow), `Analyzed` (green), `Live` (red pulse)
- If analyzed: final score + event count
- Team names (Team A vs Team B)
- Action buttons based on status:
  - Calibrated: "Analyze Video" + "Go Live"
  - Analyzed: click to view results

### API Integration

- `GET /matches` — new backend endpoint that lists all match IDs by scanning `data/matches/` directory. Returns `{matches: [{match_id, match_name, status}]}`. **This requires a small backend addition** (single endpoint, ~10 lines).
- `GET /match/{id}` — fetch full match data for each card
- Refresh on return to dashboard

---

## Section 3: Match Setup Page

### Form Fields

| Field | Type | Default | Maps to API |
|-------|------|---------|-------------|
| Match Name | text input | "Match" | `match_name` |
| Format | toggle: Best of 3 / Best of 1 | Best of 3 | `format` (`"best_of_3"` or `"best_of_1"`) |
| Deuce Rule | toggle: Golden Point / Advantage | Golden Point | `golden_point` (boolean) |
| Team A Player 1 | text input | "Player 1" | `players.P1` |
| Team A Player 2 | text input | "Player 2" | `players.P2` |
| Team B Player 3 | text input | "Player 3" | `players.P3` |
| Team B Player 4 | text input | "Player 4" | `players.P4` |
| First Server | 4 radio buttons (P1-P4) | P1 | `first_server` via teams |

### Submit

POST to `/match/setup` with:
```json
{
  "match_name": "Semi-Final",
  "players": {"P1": "Alice", "P2": "Bob", "P3": "Charlie", "P4": "Dave"},
  "teams": {"TEAM_A": ["P1", "P2"], "TEAM_B": ["P3", "P4"]},
  "golden_point": true,
  "format": "best_of_3"
}
```

On success, save match ID to localStorage match list, redirect to `/match/:id/calibrate`.

---

## Section 4: Calibration Page

### Layout

Split view:
- **Left (2/3):** Video frame (from uploaded file or camera) with clickable overlay for placing court corners
- **Right (1/3):** 3D court preview + controls

### Corner Clicking

1. User uploads a video file or connects camera. First frame displayed.
2. Instruction overlay: "Click 4 court corners: near-left → near-right → far-right → far-left"
3. Each click places a numbered blue dot on the frame. Lines connect the dots to show the court outline.
4. Counter shows "N/4 corners set". After 4, the dashed court overlay turns solid green.
5. "Reset Corners" button clears all dots.

### 3D Preview

The existing Three.js court component from the prototype, showing:
- Green court plane (10×20m)
- Net line at y=10m
- Service box lines
- Updates in real-time as corners are placed (homography applied)

### Video Source

Toggle between:
- **Upload File:** File input, display first frame on canvas
- **Camera:** `navigator.mediaDevices.getUserMedia()`, display live feed

### Save

POST to `/match/:id/calibrate` with `{corners: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]}`.

**Important:** Send pixel coordinates in the original video frame resolution, not canvas display coordinates. If the canvas is 640px wide but the video is 1280px, scale clicks by `videoWidth / canvasWidth`.

On success, show "Calibration saved!" and two buttons:
- "Analyze Video →" (navigates to `/match/:id/analyze`)
- "Go Live →" (navigates to `/match/:id/live`)

---

## Section 5: Offline Analysis Page

### Top Bar

- "Upload Video" button (file input)
- Filename display
- Progress bar: polls `GET /analyze/status/{job_id}` every 1s, shows percentage
- Status text: "Uploading..." → "Processing... 45%" → "Complete"

### Main Area (resizable panels)

**Panel 1 — Video Player (left, flex: 2):**
- HTML `<video>` element with uploaded file as source
- Scoreboard overlay at top center (translucent dark background)
  - Team A score — Team B score
  - Shows point score at the current video timestamp (interpolated from events)
- Timeline scrubber at bottom
  - Standard video controls (play/pause, seek)
  - Event markers as colored dots on the timeline (green=bounce, yellow=point, purple=serve)
  - Click marker to seek to that event

**Panel 2 — Sidebar (right, flex: 1):**

Split vertically into 3 sections:

- **Score detail (fixed height ~80px):** Full score display — team names, point score, game score, set score
- **Event log (flex: 1, scrollable):** List of all events from `GET /match/:id/events`
  - Each row: timestamp, colored dot (type), description
  - Click row → seek video to that timestamp
  - Point events highlighted with yellow background
- **3D mini-court (fixed height ~120px):** Three.js court with player position dots and ball dot
  - Blue dots = Team A, red dots = Team B, yellow dot = ball
  - Ball shows trailing path (last 15 positions, fading opacity)
  - Updates based on current video timestamp using trajectory data from `GET /match/:id/trajectory`

### API Flow

1. Upload: `POST /analyze/upload?match_id={id}` with file
2. Start: `POST /analyze/start/{job_id}`
3. Poll: `GET /analyze/status/{job_id}` every 1s until `state === "complete"` or `state === "error"`. On error, show error message from response `error` field.
4. On complete: fetch results via `GET /match/:id/score`, `GET /match/:id/events`, `GET /match/:id/trajectory`

---

## Section 6: Live View Page

### Status Bar

- LIVE indicator (red dot + "LIVE" text)
- FPS + latency display (from WebSocket status messages)
- Action buttons: "Replay 30s", "Correct Score", "Stop"

### Main Area (resizable panels)

Same layout as offline analysis, but with live data:

**Panel 1 — Camera Feed (left, flex: 2):**
- `<canvas>` element rendering JPEG frames from WebSocket
- WebSocket connects to `ws://localhost:8000/live/stream`
- Receives `{type: "frame", jpeg: base64}` → decode → draw to canvas
- Scoreboard overlay (same as offline, but updated from WebSocket score messages)
- Serving indicator at bottom-left

**Panel 2 — Sidebar (right, flex: 1):**
- **Score detail:** Updates in real-time from WebSocket `{type: "score"}` messages
- **Event feed:** Auto-scrolls, newest events at top. Events arrive via WebSocket `{type: "event"}` messages.
- **3D mini-court:** Updates every frame from WebSocket data (player positions + ball position)

### Controls

- **Replay 30s:** `GET /live/replay` → opens a modal with replay frames as a slideshow/video
- **Correct Score:** Opens a small dialog — "Award point to:" [Team A] [Team B]. Sends WebSocket `{type: "correct", team: 1|2}`.
- **Re-assign Player:** Right-click a player dot on 3D court → dialog to reassign. Sends WebSocket `{type: "reassign", track_id, player_id}`.
- **Stop:** `POST /live/stop` → redirects to `/match/:id/analyze` to see recorded results

### Startup Sequence

1. On mount: call `POST /live/start` with `{match_id, device_id: 0}` to initialize the LiveManager
2. On success: connect WebSocket to `ws://localhost:8000/live/stream`
3. If `/live/start` fails (e.g., camera not available): show error, stay on page with retry button

### WebSocket Lifecycle

1. On connect: start rendering frames
2. On message: dispatch by `type` field (frame, score, event)
3. On disconnect: show "Reconnecting..." overlay, auto-retry every 2s. On reconnect, fetch `GET /match/:id/score` and `GET /match/:id/events` (last 10) as catch-up.
4. On unmount: close WebSocket, call `POST /live/stop`

### Score Correction & Player Reassignment

Primary path: send via WebSocket (`{type: "correct", team}` or `{type: "reassign", track_id, player_id}`).
Fallback: if WebSocket is disconnected, use REST endpoints `POST /match/:id/correct-score` and `POST /match/:id/assign-player`.

---

## Section 7: Shared Components

### Component List

| Component | Used In | Purpose |
|-----------|---------|---------|
| `NavBar` | All pages | Top navigation bar |
| `Scoreboard` | Analysis, Live | Score display (overlay and sidebar versions) |
| `EventLog` | Analysis, Live | Scrollable event list with click-to-seek |
| `CourtMiniMap` | Analysis, Live, Calibration | 3D Three.js court with player/ball dots |
| `MatchCard` | Dashboard | Match summary card |
| `PanelLayout` | Analysis, Live | Resizable panel container |
| `VideoPlayer` | Analysis | HTML5 video with overlay |
| `CameraFeed` | Live | Canvas rendering WebSocket JPEG frames |
| `CalibrationCanvas` | Calibration | Clickable canvas for placing corner dots |

### API Client

Single `api.ts` module with typed functions:

```typescript
const API = "http://localhost:8000";

export async function listMatches(): Promise<MatchSummary[]> { ... }
export async function createMatch(data: MatchSetupData): Promise<{match_id: string}> { ... }
export async function getMatch(id: string): Promise<MatchData> { ... }
export async function calibrate(id: string, corners: number[][]): Promise<void> { ... }
export async function uploadVideo(matchId: string, file: File): Promise<{job_id: string}> { ... }
export async function startAnalysis(jobId: string): Promise<void> { ... }
export async function getAnalysisStatus(jobId: string): Promise<AnalysisStatus> { ... }
export async function getScore(id: string): Promise<ScoreData> { ... }
export async function getEvents(id: string): Promise<EventData[]> { ... }
export async function getTrajectory(id: string): Promise<TrajectoryPoint[]> { ... }
export async function startLive(data: LiveStartData): Promise<void> { ... }
export async function stopLive(): Promise<void> { ... }
export async function getReplay(): Promise<ReplayFrame[]> { ... }
```

### WebSocket Hook

`useWebSocket.ts` — custom React hook:

```typescript
function useWebSocket(url: string) {
  // Returns: { connected, lastFrame, lastScore, events, send, fps, latency }
  // Handles: auto-reconnect, message parsing by type, frame rendering
}
```

---

## Section 8: File Structure

```
frontend/src/
├── index.tsx                    # Entry point with BrowserRouter
├── App.tsx                      # Route definitions
├── api.ts                       # Backend API client
├── hooks/
│   └── useWebSocket.ts          # WebSocket connection hook
├── components/
│   ├── NavBar.tsx               # Top navigation
│   ├── Scoreboard.tsx           # Score display (overlay + sidebar variants)
│   ├── EventLog.tsx             # Scrollable event list
│   ├── CourtMiniMap.tsx         # 3D court with Three.js
│   ├── MatchCard.tsx            # Dashboard match card
│   ├── PanelLayout.tsx          # Resizable panel wrapper
│   ├── VideoPlayer.tsx          # HTML5 video with overlays
│   ├── CameraFeed.tsx           # Canvas for WebSocket JPEG frames
│   └── CalibrationCanvas.tsx    # Clickable corner placement overlay
├── pages/
│   ├── Dashboard.tsx            # Match list landing
│   ├── MatchSetup.tsx           # New match form
│   ├── Calibration.tsx          # Court corner calibration
│   ├── OfflineAnalysis.tsx      # Upload, analyze, view results
│   └── LiveView.tsx             # Real-time camera + scoring
├── types.ts                     # TypeScript interfaces for API data
└── styles/
    └── global.css               # Minimal global styles
```

### Existing Files

- `frontend/src/components/SetupDashboard.tsx` — legacy prototype, delete
- `frontend/src/App.tsx` — replace entirely with new route setup (salvage Three.js court code for `CourtMiniMap`)
