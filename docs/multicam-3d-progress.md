# Multi-Camera 3D Ball Tracking — Progress & Next Actions

_Status as of 2026-06-19. Branch: `feat/eval-harnesses-3d-z`._

> **2026-07-18 addendum — single-camera production path shipped.**
> - The product direction is now single-camera: `cv/monocular_trajectory.py`
>   (gravity-constrained ballistic fit) is wired into `VideoAnalyzer` and
>   verified live on real Panasonic footage (fit confidence 0.97 windows,
>   plausible 0–5 m heights). Multi-camera triangulation remains the
>   validation harness, not the product.
> - `scripts/eval_ball_labels.py` now reads the multi-source combined label
>   set (`data/labels/padelvic_panasonic_combined/`, 197 reviewed labels,
>   group-safe train/val/test) and supports `--split`. Measured @15px on the
>   held-out test rally: `tracknet_padel.pt` conf 0.3 → P/R 63.5 %;
>   conf 0.5 → P 70 % / R 53.9 % (rejected — recall feeds the ballistic fit).
>   `tracknet_phase1.pt` collapsed (0 %) and must not be used.
> - Panasonic ground homography is reproducible from the GT xlsx
>   (Positions sheet pixel↔metre pairs → RANSAC homography → project the 12
>   court keypoints); a 12-keypoint `CameraModel` PnP solves from those
>   projected keypoints, which is what enables `pixel_ray` and the monocular
>   fit on this footage.
> - `BallTracker` no longer crashes when running uncalibrated — it stays in
>   pixel space (regression-tested).

> **2026-07-16 correction:** PadelVic's synthetic CSV coordinates are Xsens
> positional ground truth and are not documented ball centers. All synthetic
> “ball accuracy” figures below are invalid and retained only as historical
> debugging notes. The scripts now require reviewed real ball labels for ball
> metrics.

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
- `scripts/eval_synthetic.py` — legacy positional-target diagnostic; refuses
  known PadelVic CSVs by default because they are not ball labels.
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

## Latest implementation

**Temporal continuity + clean 3D track** (`scripts/triangulate_matched.py`):
- Candidate-pair selection now scores reprojection error plus distance from the previous 3D ball point, so equally plausible pairs prefer continuous motion over teleports.
- A lightweight 3D track cleaner rejects physically impossible jumps using a max-speed gate and now only interpolates short gaps by default, so sparse wrong detections do not become long fake rally lines.
- Colour+motion candidates now reject elongated/low-fill blobs and tiny edge fragments using circularity, fill ratio, aspect ratio, and radius gates.
- `--debug-dir` writes side-by-side camera sheets showing raw candidates and selected pairs.
- Output JSON now contains both `raw_points` and cleaned `points`; downstream scripts use `points`.

**3D scoring harness update** (`scripts/score_from_3d.py`):
- Normal `--mode serve` is unchanged and still validates serves through the existing state machine.
- New `--mode rally` starts directly in rally state for offline 3D-track evaluation when player/server context is incomplete. This is for debugging 3D bounce/point-end behavior, not a replacement for live scoring rules.

**Detector/model finding:**
- `scripts/debug_ball_detectors.py` overlays YOLO sports-ball and TrackNet detections for sampled frames.
- On checked PADELVIC frames, generic YOLO returned no sports-ball detections; TrackNet sometimes selected wrong locations or missed entirely. This supports the suspicion that the model-based ball detector is not reliable enough for this footage.
- YOLO person boxes can help identify player/racket regions, but naive suppression is too aggressive because YOLO also detects the wall mural as a person. `triangulate_matched.py --exclude-players` is opt-in for diagnostics, not a default tracking fix.

**2D ball-first evaluation workflow:**
- `scripts/prepare_ball_label_set.py` samples real Panasonic frames into a browser labeler and starter `labels.json`.
- `scripts/eval_ball_labels.py` evaluates real-frame labels with precision/recall, misses, false positives, and pixel error.
- `scripts/eval_ball_labels.py` is the authoritative ball-coordinate gate.
- `scripts/benchmark_ball_models.py` compares ready weights on reviewed label JSON.
- Baseline on `001-1250-17462.mkv`, first 100 synthetic frames:
  - YOLO sports-ball: 0.0% detection rate.
  - Fast motion detector: 5.0% detection rate, 122.74 px median error, 0.0% within 50 px.
  - `tracknet_padel.pt`: 9.0% detection rate, 129.56 px median error, 0.0% within 50 px.
  - `tracknet_tennis.pt`: now load-compatible through a legacy 3-frame TrackNet adapter, but bad on this clip: 98.0% detection rate, 352.91 px median error, 0.0% within 50 px. It fires often, but at the wrong place.
  - `tracknetv2.pt`: present locally, but still incompatible with the implemented TrackNet architectures; its conv/batch-norm tensor shapes do not match the current loader.
  These numbers confirm the current model stack is not accurate enough.

**Molmo2/VLM probe:**
- Hugging Face currently lists AllenAI Molmo2 models including `allenai/Molmo2-4B`, `allenai/Molmo2-8B`, and `allenai/Molmo2-VideoPoint-4B`.
- `scripts/vlm_ball_probe.py` was added to run a Molmo2-style video-pointing prompt and parse `<points coords="...">` outputs into pixel coordinates. It now scores only against reviewed v1 ball-label JSON; the PadelVic synthetic CSV is not ball ground truth.
- Installed the model-card dependency stack into `backend/venv`: `transformers==4.57.1`, `accelerate`, `einops`, `decord2`, and `molmo_utils`.
- Attempted `allenai/Molmo2-VideoPoint-4B` on a 2 s 640x360 synthetic clip. The run downloaded remote processor/model code, then stalled at Hugging Face `snapshot_download` for four checkpoint shards (`0%` after 7m41s), so no inference result was produced locally.
- Practical conclusion: Molmo2 is not yet a local ball-detector answer. It may be useful later for sparse semantic labels, but ball tracking still needs a specialized detector/tracker or fine-tuning.

**Single-camera and multi-view diagnostics:**
- `scripts/annotate_single_cam_ball.py` renders one camera at a time with all colour+motion candidates in yellow and the continuity-selected 2D candidate in red.
- `scripts/annotate_multiview_ball.py` renders synchronized side-by-side camera views with each camera annotated independently. This isolates 2D detection quality before triangulation.
- Fresh 20 s diagnostics from 2026-06-19:
  - Panasonic: 1000 frames, 983 selected frames, 4.39 candidates/frame.
  - GoPro: 1199 frames, 1198 selected frames, 7.34 candidates/frame.
  - Samsung: 1200 frames, 586 selected frames, 0.95 candidates/frame.
  - iPhone: 1199 frames, 715 selected frames, 1.03 candidates/frame.
  - Panasonic+GoPro side-by-side: 500 frames at 25 fps, Panasonic selected 492 frames, GoPro selected 500 frames.
- Visual conclusion: ball tracking is currently bad before triangulation. Panasonic/GoPro often select player/racket/glass/fence/reflection blobs. Samsung/iPhone produce fewer candidates but are oblique fence views and are not calibrated for 3D.

**Verification on PADELVIC smoke windows:**
- 5 s @ 10 Hz: 5 raw matches → 18 clean points; scoring consumed the file.
- 20 s @ 10 Hz: 37 raw matches → 71 clean points; z range −0.18 … 4.47 m; x/y spread 4.9 × 16.1 m.
- After stricter blob filtering and short-gap interpolation: 20 s @ 10 Hz produced 17 raw matches → 18 clean points; far fewer fake interpolated points, but remaining selected points can still be racket/reflection false positives.
- `--mode rally` on the 20 s output emitted real events: 14 `WALL_HIT`, 1 `BOUNCE`.
- `eval_rallies.py` still reports 0 predicted rallies in that window because it grades `SERVE → POINT_END` intervals, and the current 3D-only harness does not yet produce a complete point-end pair.

## Previous breakthrough
**Colour + motion candidates + cross-camera epipolar matching** (`scripts/triangulate_matched.py`):
- per camera: ball candidates = MOVING (frame-diff) ∩ ball-COLOURED (optic-yellow HSV) ∩ small/round → ~5–9 candidates (down from 59 raw colour blobs).
- cross-camera: among candidate pairs, keep the one whose sightlines intersect (low reprojection) and lands on-court.
- **Result:** 149 matched points (was 6), ball spans the FULL court (x,y 9.1 × 21.2 m, was stuck on a head), height median 1.56 m, range −0.29 … 4.54 m = real arcs/bounces/lobs.

**Old caveat:** per-frame matching jumped to wrong yellow blobs when the ball was missed. The new continuity and clean-track pass reduces this, but more tuning is still needed.

## Next actions (in order)
1. **Improve single-camera 2D detection first.** Current model-based detection is unreliable and colour+motion admits racket/player/glass/fence/reflection false positives.
2. **Label real Panasonic ball centers** with `prepare_ball_label_set.py`, including absent/uncertain frames and hard negatives.
3. **Use `eval_ball_labels.py` as the gate** for any detector change; do not return to multi-camera 3D until 2D precision/recall is acceptable.
4. **Tune HSV to confirmed ball pixels** and add negative examples for racket/shoes/reflections.
5. **Fine-tune a padel-specific detector** if colour+motion cannot hit target accuracy.
6. **Upgrade the lightweight cleaner to a proper Kalman/ballistic tracker.** Add gravity-aware prediction, confidence scoring, and better gap handling.
7. Later: improve rally segmentation, true intrinsics, calibrate samsung/iphone for 3+ cameras, VLM semantic layer.

## How to reproduce the current best
```bash
PY=backend/venv/bin/python
# (calibrations already at /tmp/panasonic_cammodel2.json, /tmp/gopro_cammodel.json; sync at /tmp/sync_pana_gopro.json)
$PY scripts/triangulate_matched.py --start 2 --window 40 --rate 25 --max-reproj 15 --out /tmp/tri_matched.json
$PY scripts/triangulate_matched.py --start 2 --window 20 --rate 10 --max-reproj 15 \
   --debug-dir /tmp/tri_matched_debug --debug-every 10 --out /tmp/tri_matched_debug.json
$PY scripts/debug_ball_detectors.py --times 4.5,9.7,12.6,13.6,18.4 --out-dir /tmp/ball_detector_debug
$PY scripts/prepare_ball_label_set.py --start 2 --end 60 --step 10 \
   --out-dir data/labels/padelvic_panasonic_ball
$PY scripts/eval_ball_labels.py --labels data/labels/padelvic_panasonic_ball/labels.json \
   --detector color_motion --threshold-px 15 --out /tmp/ball_eval_color_motion.json
$PY scripts/benchmark_ball_models.py --max-frames 100 --out-dir /tmp/ball_model_benchmark
$PY scripts/score_from_3d.py --points /tmp/tri_matched.json --mode rally --out /tmp/score_3d_rally.json
$PY scripts/view_3d.py --points /tmp/tri_matched.json --out /tmp/matched_3d
$PY scripts/annotate_3d.py --points /tmp/tri_matched.json \
   --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
   --calib /tmp/panasonic_cammodel2.json --start 2 --window 30 --out /tmp/ball_matched.mp4
$PY scripts/annotate_single_cam_ball.py --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
   --start 2 --window 20 --width 1280 --out /tmp/single_cam_panasonic_ball_overlay.mp4
$PY scripts/annotate_multiview_ball.py \
   --video-a data/datasets/padelvic/cameras/panasonic_final.mp4 \
   --video-b data/datasets/padelvic/cameras/gopro.mp4 \
   --sync /tmp/sync_pana_gopro.json --start 2 --window 20 --rate 25 \
   --height 540 --out /tmp/multiview_panasonic_gopro_ball_overlay.mp4
```
