# Multi-Camera 3D Ball Tracking — Progress & Next Actions

_Status as of 2026-06-17. Branch: `feat/eval-harnesses-3d-z`._

## The problem we were chasing
The scoring pipeline produced **nothing** on real footage. Traced the failure down link by link:

```
scoreboard dead
  ← no rallies detected
    ← no valid serves
      ← no bounces detected
        ← ball HEIGHT (z) always 0
          ← a single camera cannot measure height (geometry, proven)
```

`BounceDetector` keys off ball height crossing a threshold; with z≡0 it never fires → serves time out → no rally → scoring engine inert. **Ball height is load-bearing for the whole system.**

## What we built (all committed)

**Evaluation harnesses (grade the analyzer against PADELVIC ground truth):**
- `backend/cli_analyze.py` — run the pipeline standalone → results.json + annotated video.
- `scripts/eval_synthetic.py` — ball-detection accuracy vs synthetic CSV GT.
- `scripts/eval_players.py` — player-position accuracy vs xlsx GT (court metres).
- `scripts/eval_rallies.py` — rally/point detection vs Plays-sheet GT.

**Multi-camera 3D pipeline:**
- `backend/src/cv/triangulation.py` (+ tests) — DLT triangulation, validated 3.5 cm @ 2 px noise.
- `scripts/sync_cameras.py` — time-align cameras (audio dead in PADELVIC → motion-energy cross-correlation; offset ≈ 0).
- `scripts/draw_calibration.py` — overlay a camera's court model on its frame (calibration check).
- `scripts/gopro_grid.py` + `scripts/calib_gopro.py` — manual gopro calibration from court corners.
- `scripts/view_3d.py` — 3D reconstruction view (court + ball trajectory).
- `scripts/annotate_3d.py` — draw the triangulated ball + height onto the video.
- `scripts/score_from_3d.py` — feed triangulated 3D into the existing scoring brain.
- `scripts/triangulate_ball.py`, `track_then_triangulate.py`, `triangulate_matched.py` — three triangulation strategies (see below).
- `scripts/ball_motion_color.py`, `ball_color_probe.py` — colour+motion ball candidates.
- `run_debug.sh`, `run_score.sh`, `run_full.sh` — one-shot runners.

**Fixes to the codebase:**
- `analyze_video` gained `max_frames`; annotated-video codec fallback avc1→mp4v.
- `YoloPlayerDetector` conf 0.6→0.4 (far-court detection 28%→68%, no accuracy loss).
- `CameraModel.project_to_ground` uses the exact homography over the PnP pose.
- `CameraModel.projection_matrix()` added for triangulation.
- `EventDetector` FAULT events carry the fault reason.

## What we learned (key findings)
1. **Single camera cannot give airborne ball position.** Tried project_to_height (z-noise → off-court) and ground-only (airborne ball projects off-court, 88%). Geometric ceiling, not tuning.
2. **Triangulation works** (3.5 cm in tests; real footage gives real height).
3. **Calibration silently poisons 3D.** gopro's auto-detected court was badly off → fixed manually. Lesson: a good homography overlay ≠ a good PnP pose (both cameras use a *guessed* focal length).
4. **Ball detection was THE blocker** — tracknet/yolo confuse a player's bald HEAD with the ball (head height z≈2.1 m). Geometry/sync/tracking all worked; the eyes were on the wrong object.
5. **Footage:** PADELVIC, one real match. Cameras used: `panasonic_final.mp4` (GT-calibrated) + `gopro.mp4` (manually calibrated), synced ≈ 0 s. samsung/iphone are oblique fence views, not calibrated. gopro's first download was truncated (re-fetched full 5.3 GB).

## The breakthrough (latest)
**Colour + motion candidates + cross-camera epipolar matching** (`scripts/triangulate_matched.py`):
- per camera: ball candidates = MOVING (frame-diff) ∩ ball-COLOURED (optic-yellow HSV) ∩ small/round → ~5–9 candidates (down from 59 raw colour blobs).
- cross-camera: among candidate pairs, keep the one whose sightlines intersect (low reprojection) and lands on-court.
- **Result:** 149 matched points (was 6), ball spans the FULL court (x,y 9.1 × 21.2 m, was stuck on a head), height median 1.56 m, range −0.29 … 4.54 m = real arcs/bounces/lobs.

**Caveat:** still raw/noisy — the matcher picks the best candidate *per frame independently*, so it jumps to wrong yellow blobs (shoe, logo, reflection) when the ball is missed. Many annotated-video markers are wrong.

## Next actions (in order)
1. **Temporal continuity in the matcher** — prefer the candidate nearest the previous 3D point (the ball can't teleport). Should kill most wrong jumps. _Highest leverage, cheap._
2. **3D ball tracker** — Kalman on (x, y, z) with a gravity/constant-velocity model; reject physically-impossible picks; interpolate gaps → clean dense flight path.
3. **Feed the clean 3D into scoring** (`score_from_3d.py`) → re-run `eval_rallies.py`. The payoff: do bounces fire → serves validate → rallies match the ground truth?
4. **Tune HSV to the actual ball colour** (sample a confirmed ball pixel) to reduce false candidates.
5. **True camera intrinsics** (not guessed focal) for tighter triangulation; optionally net-top points for metric height.
6. Later: calibrate samsung/iphone for 3+ cameras; fine-tune a padel-specific ball detector; VLM semantic layer (shot type / commentary — NOT ball detection).

## How to reproduce the current best
```bash
PY=backend/venv/bin/python
# (calibrations already at /tmp/panasonic_cammodel2.json, /tmp/gopro_cammodel.json; sync at /tmp/sync_pana_gopro.json)
$PY scripts/triangulate_matched.py --start 2 --window 40 --rate 25 --max-reproj 15 --out /tmp/tri_matched.json
$PY scripts/view_3d.py --points /tmp/tri_matched.json --out /tmp/matched_3d
$PY scripts/annotate_3d.py --points /tmp/tri_matched.json \
   --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
   --calib /tmp/panasonic_cammodel2.json --start 2 --window 30 --out /tmp/ball_matched.mp4
```
