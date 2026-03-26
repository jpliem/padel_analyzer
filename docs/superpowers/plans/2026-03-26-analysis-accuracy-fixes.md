# Analysis Accuracy & Display Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix flickering and inaccurate data display by validating data at source, storing actual video FPS, interpolating positions in the frontend, and adding calibration quality feedback.

**Architecture:** Backend validates trajectory/positions before storing (NaN, bounds). Backend stores actual video FPS in results. Frontend reads FPS, interpolates between frames instead of snapping to closest, smooths player positions. Calibration endpoint returns reprojection error for quality feedback.

**Tech Stack:** Python (backend), TypeScript/React (frontend), OpenCV (reprojection), pytest

**Spec:** `docs/superpowers/specs/2026-03-26-analysis-accuracy-fixes-design.md`

---

## File Structure

**Modified files:**
- `backend/src/cv/ball_tracker.py` — add validation before trajectory append
- `backend/src/cv/camera_model.py` — add `compute_reprojection_error()` method
- `backend/src/pipeline/video_analyzer.py` — store FPS in results, filter player positions
- `backend/main.py` — include FPS in results.json, return reprojection error from calibrate
- `frontend/src/pages/OfflineAnalysis.tsx` — read FPS, interpolation, smoothing
- `frontend/src/pages/Calibration.tsx` — display calibration quality warning
- `frontend/src/api.ts` — update calibrate return type

**Test files:**
- `backend/tests/test_ball_tracker.py` — NaN/bounds filtering tests
- `backend/tests/test_camera_model_adapter.py` — reprojection error tests

---

## Task 1: BallTracker — Validate Trajectory Points

**Files:**
- Modify: `backend/src/cv/ball_tracker.py:64-68,82-84`
- Test: `backend/tests/test_ball_tracker.py`

- [ ] **Step 1: Write failing tests for NaN and bounds filtering**

```python
# Add to backend/tests/test_ball_tracker.py
import math

def test_nan_position_not_added_to_trajectory():
    """Trajectory should skip NaN positions."""
    from cv.ball_tracker import BallTracker
    from unittest.mock import MagicMock
    cal = MagicMock()
    cal.project_to_ground.return_value = (float('nan'), float('nan'))
    cal.has_3d = MagicMock(return_value=False)
    tracker = BallTracker(calibration=cal)
    tracker._initialized = True
    tracker._kf.x = [float('nan'), float('nan'), 0, 0]
    tracker._kf.predict = MagicMock()
    tracker._kf.update = MagicMock()
    # Feed a detection that projects to NaN
    result = tracker.update([100, 100, 120, 120], frame_number=1)
    # Should either return None or a valid position — never NaN in trajectory
    for pos in tracker.trajectory:
        assert not math.isnan(pos["x"])
        assert not math.isnan(pos["y"])


def test_out_of_bounds_position_not_added():
    """Positions more than 30m from court center should be skipped."""
    from cv.ball_tracker import BallTracker
    from unittest.mock import MagicMock
    cal = MagicMock()
    cal.project_to_ground.return_value = (100.0, 200.0)  # Way off court
    cal.has_3d = MagicMock(return_value=False)
    tracker = BallTracker(calibration=cal)
    tracker._initialized = True
    tracker._kf.x = [100.0, 200.0, 0, 0]
    tracker._kf.predict = MagicMock()
    tracker._kf.update = MagicMock()
    result = tracker.update([100, 100, 120, 120], frame_number=1)
    # Should not be in trajectory
    for pos in tracker.trajectory:
        dist = ((pos["x"] - 5) ** 2 + (pos["y"] - 10) ** 2) ** 0.5
        assert dist < 30, f"Position ({pos['x']}, {pos['y']}) is {dist}m from center"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && source venv/bin/activate && python -m pytest tests/test_ball_tracker.py::test_nan_position_not_added_to_trajectory tests/test_ball_tracker.py::test_out_of_bounds_position_not_added -v`

- [ ] **Step 3: Add validation to BallTracker.update()**

In `backend/src/cv/ball_tracker.py`, add a validation helper and use it before both trajectory appends:

```python
# Add after line 32 (end of __init__)
def _is_valid_position(self, x: float, y: float, z: float) -> bool:
    """Check if position is valid (not NaN, within reasonable bounds)."""
    import math
    if math.isnan(x) or math.isnan(y) or math.isnan(z):
        return False
    # More than 30m from court center (5, 10) is clearly wrong
    if (x - 5) ** 2 + (y - 10) ** 2 > 900:  # 30^2
        return False
    return True
```

Then wrap the trajectory append at line 68:
```python
# Replace line 68: self.trajectory.append(pos)
if self._is_valid_position(pos["x"], pos["y"], pos["z"]):
    self.trajectory.append(pos)
```

And wrap the predicted position append at line 84:
```python
# Replace line 84: self.trajectory.append(pos)
if self._is_valid_position(pos["x"], pos["y"], pos["z"]):
    self.trajectory.append(pos)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/backend && python -m pytest tests/test_ball_tracker.py -v`

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest --tb=short -q`

- [ ] **Step 6: Commit**

```bash
git add backend/src/cv/ball_tracker.py backend/tests/test_ball_tracker.py
git commit -m "fix: filter NaN and out-of-bounds positions from ball trajectory"
```

---

## Task 2: CameraModel — Reprojection Error

**Files:**
- Modify: `backend/src/cv/camera_model.py`
- Test: `backend/tests/test_camera_model_adapter.py`

- [ ] **Step 1: Write failing test**

```python
# Add to backend/tests/test_camera_model_adapter.py
def test_compute_reprojection_error(calibrated_camera):
    """Reprojection error should be a positive number for calibrated camera."""
    keypoints_2d = [
        [100, 600], [900, 600],
        [200, 450], [500, 450], [800, 450],
        [250, 350], [750, 350],
        [200, 250], [500, 250], [800, 250],
        [100, 100], [900, 100],
    ]
    error = calibrated_camera.compute_reprojection_error(keypoints_2d)
    assert error is not None
    assert error >= 0.0
    # For a well-calibrated camera, error should be reasonable
    assert error < 100.0  # pixels


def test_reprojection_error_uncalibrated():
    """Uncalibrated camera should return None."""
    from cv.camera_model import CameraModel
    cam = CameraModel()
    error = cam.compute_reprojection_error([[100, 200]])
    assert error is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_camera_model_adapter.py::test_compute_reprojection_error tests/test_camera_model_adapter.py::test_reprojection_error_uncalibrated -v`

- [ ] **Step 3: Implement compute_reprojection_error()**

Add to `backend/src/cv/camera_model.py` in the `CameraModel` class:

```python
def compute_reprojection_error(self, keypoints_2d) -> Optional[float]:
    """Compute mean reprojection error in pixels."""
    if self.rvec is None or self.tvec is None:
        return None
    import numpy as np
    # Use the same 3D reference points used during calibration
    pts_3d = []
    pts_2d = []
    n = len(keypoints_2d)
    ref_3d = GROUND_KEYPOINTS_3D[:n]
    for i, (kp, ref) in enumerate(zip(keypoints_2d, ref_3d)):
        pts_3d.append(ref)
        pts_2d.append(kp)
    if not pts_3d:
        return None
    pts_3d = np.array(pts_3d, dtype=np.float64)
    pts_2d = np.array(pts_2d, dtype=np.float64)
    projected, _ = cv2.projectPoints(
        pts_3d, self.rvec, self.tvec,
        self.camera_matrix, self.dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    errors = np.sqrt(np.sum((projected - pts_2d) ** 2, axis=1))
    return float(np.mean(errors))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_camera_model_adapter.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/src/cv/camera_model.py backend/tests/test_camera_model_adapter.py
git commit -m "feat: compute_reprojection_error for calibration quality feedback"
```

---

## Task 3: Backend — Store FPS and Reprojection Error

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/src/pipeline/video_analyzer.py`

- [ ] **Step 1: Store FPS in analyze_video() return value**

In `backend/src/pipeline/video_analyzer.py`, update the return dict at line 202:

```python
return {
    "match_id": self._match_id,
    "frames_processed": frame_no,
    "events": len(self.all_events),
    "final_score": self.scoring_engine.get_score_display(),
    "fps": fps,
}
```

- [ ] **Step 2: Add FPS to results.json in main.py**

In `backend/main.py`, in both results.json write locations:

In `start_match_analysis` (around line 484), add `"fps"` to the JSON dump:
```python
"fps": result.get("fps", 30.0),
```

In `start_analysis` (around line 664), same addition:
```python
"fps": result.get("fps", 30.0),
```

- [ ] **Step 3: Return reprojection error from calibrate endpoint**

In `backend/main.py`, in the `calibrate_court` function (around line 247), after calibration:

```python
reproj_error = cam.compute_reprojection_error(req.corners)
match_data["reprojection_error"] = reproj_error
```

And update the return value to include it:
```python
return {"status": "calibrated", "match_id": match_id, "mode": mode,
        "reprojection_error": reproj_error}
```

- [ ] **Step 4: Filter player positions in VideoAnalyzer**

In `backend/src/pipeline/video_analyzer.py`, update the player positions log (lines 148-155):

```python
# Filter invalid player positions before logging
valid_players = [
    p for p in player_pos
    if not (p["x"] == 0 and p["y"] == 0)  # failed projection
    and -5 <= p["x"] <= 15 and -5 <= p["y"] <= 25  # within extended bounds
][:4]  # cap at 4 players

self.player_positions_log.append({
    "frame": frame_no,
    "players": [
        {"track_id": p["track_id"], "x": p["x"], "y": p["y"],
         "player_id": self.player_tracker.get_player_id(p["track_id"])}
        for p in valid_players
    ],
})
```

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest --tb=short -q`

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/src/pipeline/video_analyzer.py
git commit -m "feat: store FPS in results, return reprojection error, filter player positions"
```

---

## Task 4: Frontend — Read Actual FPS

**Files:**
- Modify: `frontend/src/pages/OfflineAnalysis.tsx`
- Modify: `frontend/src/api.ts`

- [ ] **Step 1: Add FPS to analysis status response**

In `frontend/src/api.ts`, add a new function to get results metadata:

```typescript
export const getResultsMeta = async (matchId: string): Promise<{ fps: number }> => {
  const resp = await fetch(`${API}/match/${matchId}/trajectory`);
  const data = await resp.json();
  // FPS comes from the trajectory endpoint or we check the match config
  return { fps: 30 }; // fallback
};
```

Actually, simpler approach — read FPS from the analysis status endpoint. In `backend/main.py`, update `get_analysis_status` to include FPS from results if available:

In `backend/main.py`, update the status endpoint to include FPS:
```python
@app.get("/analyze/status/{job_id}")
def get_analysis_status(job_id: str):
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _analysis_jobs[job_id]
    # Include FPS from results if analysis is complete
    if job.get("state") == "complete":
        results = _load_results(job["match_id"])
        if results:
            job["fps"] = results.get("fps", 30.0)
    return job
```

- [ ] **Step 2: Update OfflineAnalysis to use actual FPS**

In `frontend/src/pages/OfflineAnalysis.tsx`:

Change line 27 from:
```typescript
const [fps] = useState(24);
```
To:
```typescript
const [fps, setFps] = useState(30);
```

In the `loadResults` callback (after data is loaded), read FPS from status:
```typescript
const loadResults = useCallback(async () => {
    if (!id) return;
    const [sd, ed, td, pd] = await Promise.all([
      getScore(id), getEvents(id), getTrajectory(id), getPositions(id),
    ]);
    setScore(sd);
    setEvents(ed.events || []);
    setTrajectory(td.trajectory || []);
    setPositions(pd.positions || []);
    // Try to get FPS from status
    try {
      const st = await getAnalysisStatus(id);
      if (st.fps) setFps(st.fps);
    } catch {}
    setStatus('complete');
  }, [id]);
```

- [ ] **Step 3: Run frontend build to check for errors**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm run build 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/OfflineAnalysis.tsx backend/main.py
git commit -m "feat: read actual video FPS instead of hardcoded 24"
```

---

## Task 5: Frontend — Trajectory & Position Interpolation

**Files:**
- Modify: `frontend/src/pages/OfflineAnalysis.tsx`

- [ ] **Step 1: Replace ball "closest frame" snap with interpolation**

In `frontend/src/pages/OfflineAnalysis.tsx`, replace lines 139-143:

```typescript
// OLD: const rawBall = trajectory.reduce<TrajectoryPoint | null>((closest, t) => ...

// NEW: Interpolate ball position between two bracketing frames
const interpolateBall = (traj: TrajectoryPoint[], frame: number): TrajectoryPoint | null => {
  if (traj.length === 0) return null;
  // Find bracketing points
  let before: TrajectoryPoint | null = null;
  let after: TrajectoryPoint | null = null;
  for (const t of traj) {
    if (t.frame <= frame && (!before || t.frame > before.frame)) before = t;
    if (t.frame >= frame && (!after || t.frame < after.frame)) after = t;
  }
  if (!before && !after) return null;
  if (!before) return after;
  if (!after) return before;
  if (before.frame === after.frame) return before;
  // Linear interpolation
  const ratio = (frame - before.frame) / (after.frame - before.frame);
  return {
    x: before.x + (after.x - before.x) * ratio,
    y: before.y + (after.y - before.y) * ratio,
    z: before.z + (after.z - before.z) * ratio,
    speed: before.speed + (after.speed - before.speed) * ratio,
    timestamp: before.timestamp + (after.timestamp - before.timestamp) * ratio,
    frame: frame,
    detected: before.detected && after.detected,
  };
};

const rawBall = interpolateBall(trajectory, currentFrame);
const currentBall = rawBall && isOnCourt(rawBall.x, rawBall.y) ? rawBall : null;
```

- [ ] **Step 2: Replace player position "closest frame" snap with interpolation**

Replace lines 149-158:

```typescript
// Interpolate player positions between two bracketing frames
const interpolatePlayers = (pos: FramePositions[], frame: number) => {
  if (pos.length === 0) return [];
  let before: FramePositions | null = null;
  let after: FramePositions | null = null;
  for (const p of pos) {
    if (p.frame <= frame && (!before || p.frame > before.frame)) before = p;
    if (p.frame >= frame && (!after || p.frame < after.frame)) after = p;
  }
  if (!before && !after) return [];
  if (!before) return after!.players;
  if (!after) return before.players;
  if (before.frame === after.frame) return before.players;

  const ratio = (frame - before.frame) / (after.frame - before.frame);
  // Match players by player_id or track_id between frames
  return before.players.map(bp => {
    const ap = after!.players.find(
      p => (p.player_id && p.player_id === bp.player_id) || p.track_id === bp.track_id
    );
    if (!ap) return bp;
    return {
      ...bp,
      x: bp.x + (ap.x - bp.x) * ratio,
      y: bp.y + (ap.y - bp.y) * ratio,
    };
  });
};

const interpolatedPlayers = interpolatePlayers(positions, currentFrame);

const playerDots = interpolatedPlayers.map(p => ({
  id: p.player_id || `#${p.track_id}`,
  x: p.x,
  y: p.y,
  team: (p.player_id === 'P1' || p.player_id === 'P2' ? 'A' : 'B') as 'A' | 'B',
  label: p.player_id || `#${p.track_id}`,
}));
```

- [ ] **Step 3: Run frontend build**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm run build 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/OfflineAnalysis.tsx
git commit -m "feat: interpolate ball and player positions between frames"
```

---

## Task 6: Frontend — Player Position Smoothing

**Files:**
- Modify: `frontend/src/pages/OfflineAnalysis.tsx`

- [ ] **Step 1: Add smoothing buffer and apply moving average**

Add a ref to track position history, and apply smoothing after interpolation:

```typescript
// Add near the top of the component, after other refs
const playerHistoryRef = useRef<Record<string, Array<{x: number, y: number}>>>({});

// Add after interpolatedPlayers calculation, before playerDots mapping
const smoothedPlayers = interpolatedPlayers.map(p => {
  const key = p.player_id || `#${p.track_id}`;
  if (!playerHistoryRef.current[key]) {
    playerHistoryRef.current[key] = [];
  }
  const history = playerHistoryRef.current[key];
  const last = history[history.length - 1];

  // If position jumped more than 3m, reset buffer (likely ID swap)
  if (last && Math.sqrt((p.x - last.x) ** 2 + (p.y - last.y) ** 2) > 3) {
    playerHistoryRef.current[key] = [{ x: p.x, y: p.y }];
    return p;
  }

  history.push({ x: p.x, y: p.y });
  if (history.length > 3) history.shift(); // Keep last 3

  // Moving average
  const avgX = history.reduce((s, h) => s + h.x, 0) / history.length;
  const avgY = history.reduce((s, h) => s + h.y, 0) / history.length;
  return { ...p, x: avgX, y: avgY };
});
```

Then update `playerDots` to use `smoothedPlayers` instead of `interpolatedPlayers`.

- [ ] **Step 2: Run frontend build**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm run build 2>&1 | tail -10`

- [ ] **Step 3: Commit**

```bash
git add frontend/src/pages/OfflineAnalysis.tsx
git commit -m "feat: 3-frame moving average smoothing for player positions"
```

---

## Task 7: Frontend — Calibration Quality Warning

**Files:**
- Modify: `frontend/src/pages/Calibration.tsx`
- Modify: `frontend/src/api.ts`

- [ ] **Step 1: Update api.ts calibrate to return reprojection error**

In `frontend/src/api.ts`, update the `calibrate` function to return the response:

```typescript
export const calibrate = async (
  matchId: string, corners: number[][], netPoints: number[][] | null,
  netTopPoints: number[][] | null
): Promise<{ status: string; reprojection_error?: number }> => {
  const resp = await fetch(`${API}/match/${matchId}/calibrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      corners, net_points: netPoints, net_top_points: netTopPoints,
    }),
  });
  return resp.json();
};
```

- [ ] **Step 2: Display calibration quality in Calibration.tsx**

Add state for reprojection error and display a warning:

```typescript
const [reprojError, setReprojError] = useState<number | null>(null);
```

After the calibrate call (around line 88), capture the error:
```typescript
const result = await calibrate(id, keypoints, null, netTopPoints);
setReprojError(result.reprojection_error ?? null);
```

Add a warning display after the "Saved!" message:
```typescript
{reprojError !== null && (
  <div style={{
    padding: '8px 16px',
    borderRadius: 4,
    marginTop: 8,
    background: reprojError < 10 ? '#d4edda' : reprojError < 20 ? '#fff3cd' : '#f8d7da',
    color: reprojError < 10 ? '#155724' : reprojError < 20 ? '#856404' : '#721c24',
  }}>
    {reprojError < 10
      ? `Good calibration (${reprojError.toFixed(1)}px error)`
      : reprojError < 20
        ? `Calibration may be inaccurate (${reprojError.toFixed(1)}px error) — consider re-clicking keypoints`
        : `Calibration is poor (${reprojError.toFixed(1)}px error) — results will be unreliable`}
  </div>
)}
```

- [ ] **Step 3: Run frontend build**

Run: `cd /Users/jonathan/Documents/Github/padel_analyzer/frontend && npm run build 2>&1 | tail -10`

- [ ] **Step 4: Commit**

```bash
git add frontend/src/pages/Calibration.tsx frontend/src/api.ts
git commit -m "feat: display calibration quality warning with reprojection error"
```
