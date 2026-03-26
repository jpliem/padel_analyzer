# Analysis Accuracy & Display Fixes

## Overview

Fix flickering and inaccurate data display in the padel analyzer. Root causes: hardcoded FPS assumption, missing data validation, no trajectory interpolation, no calibration quality feedback. Changes span backend (data quality) and frontend (rendering robustness).

## 1. Backend — FPS & Calibration Quality

### Store actual video FPS
- In `VideoAnalyzer.analyze_video()`, extract FPS via `cap.get(cv2.CAP_PROP_FPS)` and store in `results.json` as `"fps": <float>`.
- The analysis status/results endpoints return the FPS so the frontend can read it.

### Calibration reprojection error
- Add `compute_reprojection_error(keypoints_2d)` to `CameraModel`. Projects the known 3D court points back to pixels via `cv2.projectPoints`, computes mean pixel distance from the user-provided keypoints.
- `POST /match/{id}/calibrate` returns `"reprojection_error": <float>` in its response.
- Stored in match config (`config.json`) for later display.

## 2. Backend — Trajectory & Position Data Quality

### BallTracker validation
In `BallTracker.update()`, before appending to trajectory:
- Skip if `math.isnan(x)` or `math.isnan(y)` or `math.isnan(z)`
- Skip if position is more than 30m from court center (5, 10): `sqrt((x-5)^2 + (y-10)^2) > 30`
- Existing `detected` flag already distinguishes real detections from Kalman predictions — no change needed.

### Player position validation
In `VideoAnalyzer.process_frame()`, after PlayerTracker returns positions:
- Filter out positions with `x == 0 and y == 0` (failed projection)
- Filter out positions outside extended court bounds: `x < -5 or x > 15 or y < -5 or y > 25`
- Cap at 4 players per frame in the position log (not just in 3D view)

## 3. Frontend — FPS Sync

### Read actual FPS
- `OfflineAnalysis.tsx` currently hardcodes `fps = 24`.
- After analysis completes, read FPS from results. If not available, default to 30.
- Frame calculation becomes: `Math.floor(currentTime * actualFps)`

## 4. Frontend — Trajectory & Position Interpolation

### Ball interpolation
Replace "closest frame" snap with linear interpolation:
- Find the two trajectory points bracketing `currentFrame`: one with `frame <= currentFrame` and one with `frame >= currentFrame`.
- Interpolate x, y, z linearly based on frame position between them.
- If only one side exists (at start/end), use the single point.
- Result: smooth ball movement even when detection is sparse (e.g., every 3rd frame).

### Player position interpolation
Same approach for player positions:
- Find two `FramePositions` entries bracketing `currentFrame`.
- For each player (matched by `player_id` or `track_id`), interpolate x/y.
- If a player exists in one frame but not the other, use the available position without interpolation.

### Player position smoothing
Apply 3-frame moving average to rendered player positions:
- Maintain a small buffer of the last 3 interpolated positions per player.
- Render the average. This eliminates single-frame jitter.
- If a position jumps more than 3m from the previous, reset the buffer (likely a tracking ID swap, not real movement).

## 5. Frontend — Calibration Quality Warning

### On Calibration page
After `POST /match/{id}/calibrate` returns:
- Display reprojection error value.
- Error < 10px: green "Good calibration"
- Error 10-20px: yellow warning "Calibration may be inaccurate — consider re-clicking keypoints"
- Error > 20px: red warning "Calibration is poor — results will be unreliable"

### On OfflineAnalysis page
- If match config has `reprojection_error > 15`, show a banner: "Calibration quality is low — results may be inaccurate. Consider recalibrating."

## 6. Files Changed

**Backend:**
- `backend/src/pipeline/video_analyzer.py` — extract video FPS, filter invalid player positions, store FPS in results
- `backend/src/cv/ball_tracker.py` — NaN check, bounds check before appending to trajectory
- `backend/src/cv/camera_model.py` — `compute_reprojection_error()` method
- `backend/main.py` — return FPS in results endpoints, return reprojection_error from calibrate

**Frontend:**
- `frontend/src/pages/OfflineAnalysis.tsx` — read actual FPS, interpolate trajectory/positions, smooth player positions
- `frontend/src/pages/Calibration.tsx` — display calibration quality warning
- `frontend/src/components/CourtMiniMap.tsx` — handle interpolated data, dim undetected ball

**Tests:**
- `backend/tests/test_ball_tracker.py` — NaN filtering, bounds filtering
- `backend/tests/test_camera_model_adapter.py` — reprojection error computation
