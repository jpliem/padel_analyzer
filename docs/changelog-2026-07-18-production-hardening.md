# 2026-07-18 — Production hardening, single-cam verification, VLM audit layer

Everything below happened in one working session on branch `feat/eval-harnesses-3d-z`.
Backend tests: 542 → **556 passing**. Frontend builds clean.

## Bugs found and fixed

### 1. Web UI could not scroll (critical UX)
`frontend/public/index.html` shipped `body { margin: 0; background: #000; overflow: hidden; }`
from a leftover template. Every page was clipped to the viewport — the **review queue,
score card, and stats were unreachable** on normal laptop screens.
Fix: removed `overflow: hidden`, background matched to app theme. Verified headless:
`document.body` overflow now `visible`, analysis page scrolls to 1328px height.

### 2. Uncalibrated analysis crashed the pipeline
`BallTracker.update` called `pixel_to_court()` unconditionally; an uncalibrated
`CourtCalibration` raised `RuntimeError` and killed the whole analysis (contradicting
cli_analyze's promise of pixel-space fallback). Fix: `BallTracker._to_court()` guards on
`homography is None` → pixel passthrough. TDD: 2 regression tests in
`backend/tests/test_ball_tracker.py::TestUncalibratedFallback`.

### 3. Monocular fit certified off-court garbage as "reliable"
Visual audit of real-footage output caught fits stamped `reliable, confidence 0.97` at
court y = −8 (8 m behind the baseline) and z = 5+ m — geometrically consistent
wrong-object tracks (racket/reflection). Ray error and conditioning gates cannot catch
these. Fix: plausibility gate in `MonocularTrajectoryEstimator` — every fitted point
must lie inside the playable volume (x ∈ [−2, 12], y ∈ [−2, 22], z ≤ 12 m) or the fit is
unreliable with confidence 0. TDD: 3 tests in `test_single_camera_geometry.py`.
Verified on real footage: 69 monocular points in the previously-poisoned window, **zero**
off-court, z range 0.1–4.5 m.

### 4. Unknown frontend routes rendered a blank page
No catch-all route. Fix: `<Route path="*" element={<Navigate to="/" replace />} />`.

### 5. No way to stop a runaway analysis (found by user, samsung.mp4 incident)
Full-resolution TrackNet on CPU is heavy; a long video pegged the machine with no off
switch short of killing the server. Fix:
- `VideoAnalyzer.cancel_requested` flag checked every frame; result carries `cancelled`.
- `POST /analyze/cancel/{job_id}` endpoint (404 unknown job, 409 nothing running).
- "Cancel analysis" button on the processing screen; job state `cancelled` handled by
  the poller with a friendly retry message.

## API hardening (`backend/main.py`)

- **Path-traversal-safe identifiers**: `_require_safe_id()` (`^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$`,
  no `..`) enforced in `_match_dir()` and all three template path joins.
- **Upload cap**: streaming copy with `PADEL_MAX_UPLOAD_BYTES` limit (default 8 GB) → 413.
- **Error hygiene**: background-task failures log full traceback server-side
  (`logging` configured, `PADEL_LOG_LEVEL`), clients get a generic message — no more
  raw `str(e)` leaks.
- **DELETE /match/{id}** now 404s on missing match instead of pretending success.
- **Detector default** unified to `tracknet` (was `yolo` in one fallback path).
- Tests: `backend/tests/test_api.py::TestHardening` (5 cases).

## Ball-label evaluation gate repaired

`scripts/eval_ball_labels.py` (the authoritative detector gate) crashed on the combined
multi-source label set. Now:
- Resolves per-label `source_video` (multi-source v1 schema), one capture per video.
- `--split train|val|test` filter; split recorded in the summary.
- Legacy single-video schema still works (back-compat in `compute_metrics`).

Label inventory: `data/labels/padelvic_panasonic_combined/labels.json` — 197 reviewed
labels, group-safe splits (train 71 / val 70 / test 56), two cameras/sequences.

Measured @15px, held-out test rally (`panasonic_final.mp4:2361-2920`):

| Model | Precision | Recall | Notes |
|---|---|---|---|
| `tracknet_padel.pt` conf 0.3 | 63.5% | 63.5% | **production** — median error 0 px when correct |
| `tracknet_padel.pt` conf 0.5 | 70.0% | 53.9% | rejected: ballistic fit needs recall |
| `tracknet_phase1.pt` | 0% | 0% | collapsed retrain — never use |
| `tracknet_phase1_candidate.pt` | 57.6% (val) | 57.6% | not adopted |

## End-to-end verification on real footage

`cli_analyze.py` on `panasonic_final.mp4`, 3000 frames (60 s), 12-keypoint CameraModel:
- 7.1 fps processing; 2918 trajectory points, 1605 detected.
- **734 points from the monocular ballistic fit** (real single-camera height).
- Event chain fires end-to-end: 1 SERVE, 1 BOUNCE, 2 HIT, 2 WALL_HIT, 29 FAULT,
  15 POINT_END.
- Score remains 0-0 **by design**: `review_ledger.propose(auto_confirm_threshold=2.0)`
  means CV never auto-confirms a point; only human-confirmed points score.

### Panasonic calibration is reproducible
`/tmp` calibrations were lost; rebuilt from the PADELVIC GT xlsx:
1. `Positions` sheet gives pixel↔metre pairs (11 060 points) → RANSAC homography.
2. Project court corners + net posts → legacy 4-corner calib.
3. Project the 12 court keypoints → `camera_model_keypoints` → CameraModel PnP solves
   → `pixel_ray` available → monocular fit active.
Script pattern preserved in this doc; regenerating takes seconds.

## Frontend: 3D court view finally wired

`Court3DView` (Three.js) existed but had **zero importers**. Now in OfflineAnalysis
Court tab: 2D/3D toggle, ball rendered at fitted height with ground shadow + height
line, 3-second trail, players as figures, honesty caption ("Ball height from ballistic
fit; flat when the fit was unreliable"). CSS: `.court-view-toggle`, `.court-3d-shell`.

## New: VLM track-audit layer (`vlm_coach/track_audit.py`)

Rationale: metrics said "63.5%, confidence 0.97" while rendered output showed the
marker on grass. Seeing is auditing. The audit layer automates that look:

1. Samples decision frames (event frames, monocular-fit frames, detections).
2. Renders the tracker's claim (red marker + trail) onto each frame.
3. Local vision model (Ollama, default `qwen2.5vl:3b`) judges each frame against the
   `TrackAuditVerdict` schema: `marker_on_ball` yes/close/no/no_ball_visible/unclear,
   `ball_visible_elsewhere`, location hint, confidence.
4. Report: agreement stats + `label_queue` = disagreement frames → feed to
   `prepare_ball_label_set.py` for the next fine-tune round (active learning).

Verdicts are review material only — they never change the score (same policy as all
other evidence). Tests: `backend/tests/test_track_audit.py` (4, fake client; no model
needed in CI).

Architecture position:
```
CNN eyes (fast, precise, dumb)
  → physics brain (ballistic fit, padel rules)
    → VLM auditor (slow, smart, sees like a human)
      → human review queue (final authority)
```
Each layer catches the failure mode of the layer below. VLM-as-tracker was evaluated
and rejected (too slow, fails fine-grained localization — consistent with TennisTV
benchmark findings); VLM-as-auditor is the correct role.

## Full web-UI end-to-end run (proof)

Driven entirely through the browser (headless Chromium), match `bdf7d428`:
1. New-match form → attach `game2_vic_panasonic.mp4` (15 MB, 14 s) → create.
2. Auto-navigated to calibration. Auto-detect failed **with visible error** (honest UX).
3. Clicked 12 court keypoints manually; overlay aligned with real court lines; saved
   (2D-only — net-top clicks skipped, so no monocular z this run, as designed).
4. Analysis auto-started; progress screen with working Cancel button; completed ~4 min.
5. Results page: 693 track points (462 detected), 104 candidates rejected,
   46 uncertain frames, 31 contact proposals, 2 rule decisions, 1 fault review item,
   model-evidence panel, annotated video, 2D/3D court view — all rendered.

UX gaps recorded during the run:
- "Analyze Video" button exists only in the just-saved calibration panel.
- `LiveView` hardcodes `ws://localhost:8000`.
- 3D height needs the optional net-top clicks; worth a stronger prompt in the UI.

## VLM audit: live-run constraint on 8 GB machines

`track_audit.py` ran mechanically end-to-end. Model reality on this 8 GB Mac:
- `qwen2.5vl:3b` fails to load (~5 GB needed; 1.3 GB free with app stack up) —
  fails even with servers stopped when other apps hold memory.
- `moondream` (1.8B) loads but returns degenerate identical verdicts (7/7 "unclear",
  same confidence) — below task capability.
Conclusion: audit layer requires either ≥5 GB free for a 3B model, a cloud vision API
provider, or a dedicated machine. Code and tests are model-agnostic (`--model` flag).

## Operational notes

- Backend for local dev on port **8001** (8000 occupied by an unrelated app on this
  machine); frontend `REACT_APP_API_PORT=8001 PORT=3001 npm start`.
- `LiveView` still hardcodes `ws://localhost:8000/live/stream` — known leftover.
- Processing heat/speed: TrackNet CPU at 3626×1960 ≈ 2-7 fps. A full match takes hours
  and works the machine hard. Run one analysis at a time; cancel is now available.
  GPU batch inference is the planned fix.

## Remaining gaps (priority order)

1. Commit this work (large uncommitted tree on `feat/eval-harnesses-3d-z`).
2. Detector accuracy 63.5% → 85%+: TrackNetV3-style temporal rectification port,
   fine-tune on audit-flagged frames; gate every change with `eval_ball_labels.py`.
3. Auth + worker queue before any multi-user/public deployment.
4. Automated frontend E2E (Playwright) — today's audits were scripted but ad-hoc.
5. GPU inference for speed; frame-skip option for preview-quality passes.
