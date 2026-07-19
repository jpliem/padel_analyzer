# Changelog 2026-07-19 — SOTA research, auditor redesign, OSNet re-ID, repo consolidation

One-day session. Everything below is merged to `master` and covered by the
test suite (569 passing at close).

## Research (verified, multi-source)

Full sweep of 2025-2026 models applicable to this pipeline. Key conclusions:

- **Ball detection gap is enormous**: TrackNetV3 reaches 97.5% acc / 98.6 F1
  (shuttlecock) and TrackNetV3-class variants 94.5P/88.0R on tennis vs our
  63.5% P/R. TrackNetV4 adds a plug-and-play motion-attention layer. BlurBall
  shows explicit motion-blur labeling cuts trajectory MAE ~40%. Single-frame
  YOLO confirmed wrong tool (57.8% acc in the same benchmark).
- **Dataset found**: PadelTracker100 (Zenodo 14653706, CC-BY-4.0, 7.1 GB) —
  ~100k annotated 1080p frames from WPT Finals 2022: ball trajectory, player
  positions, ViTPose-L poses, 6-class shot events. First real-footage padel
  ball ground truth available to us at scale. Inter-annotator ball IoU is
  0.677 (motion blur) — ceiling to remember when reading eval numbers.
- **Player tracking**: sports MOT SOTA moved to motion-agnostic association
  (Deep-EIoU: HOTA 0.42→0.54 over ByteTrack; Deep HM-SORT 80.1 HOTA on
  SportsMOT). Lightweight OSNet re-ID beat transformer re-ID under fast
  motion. Pseudo-labeling lifted HOTA 0.380→0.491 elsewhere — cheap idea for
  our unlabeled footage.
- **Local VLM landscape vs our M2/8GB**: Molmo 2 (Dec 2025) is the best
  video-pointing model on paper but cannot run here (11-24 GB, INT4 broken);
  Qwen3.5-VL has no sub-9B variant; SAM 3 needs ~4 GB VRAM on NVIDIA at best.
  Working local choice: qwen3-vl:2b-instruct (~2 GB).

## VLM track audit — measured, then redesigned

Built `scripts/eval_vlm_auditor.py`: renders markers at ground-truth vs
deliberately displaced positions on labeled frames, scores any model's
verdicts. Results (12 frames × 2 conditions):

| Model | correct-marker acc | wrong-marker catch | behaviour |
|---|---|---|---|
| qwen3-vl:2b-instruct | 1.00 | 0.00 | says yes to everything |
| qwen2.5vl:3b | 0.08 | 0.92 | says no to everything |
| moondream | 0.00 | 0.00 | 100% "unclear" |

Judgment prompts are unusable at this model size — full frame or crop.
Pointing is not: on 320px crops from the original-resolution video,
qwen3-vl:2b-instruct pointed at the ball with ≤10px error (4/6 found, misses
are safe uncertainty). Coordinates arrive 0-1000 normalized (Qwen convention).

**Redesign shipped** in `vlm_coach/track_audit.py`: default `--mode pointing`
crops around the tracker's claim, asks the model only to locate the ball, and
derives the verdict from pointed-vs-claimed distance in code. The VLM locates;
the code judges. First live run correctly flagged 7/7 frames where the tracker
sat on a sideline barrier. Legacy behaviour kept under `--mode judgment`.

Supporting fixes: qwen3-vl *thinking* variants burn the whole `num_predict`
budget on `<think>` and return empty content under structured output — use
`-instruct` tags; `OllamaClient` num_predict raised to 2048. A stale manually
launched `ollama serve` was found owning port 11434 and half-broken (models
failed to load); killed, brew service now owns the port.

Human-in-the-loop: `scripts/review_vlm_auditor.py` renders any auditor report
as an HTML review page (who is right — model or label?) exporting overrides
JSON; `scripts/eval_vlm_pointing.py` scores pointing accuracy directly.

## Player re-ID — OSNet replaces HSV

`OsnetAppearanceEncoder` (OSNet x0.25, MSMT17 weights, vendored MIT model
file, 8.9 MB, runs on MPS) wired into `PlayerTracker` with graceful HSV
fallback. Measured on real similar-kit player crops: HSV separation −0.003
(cannot distinguish the players at all); OSNet +0.042 with hand boxes at
preview resolution. Production uses full-resolution crops, so real separation
should be larger. Full `eval_players.py` run still pending.

## Ported from abandoned `feature/accuracy-fixes` branch (March 2026)

- `CameraModel.compute_reprojection_error()` — mean pixel error of clicked
  keypoints vs fitted camera; calibrate endpoint stores and returns it;
  Calibration page shows good/warn/poor verdict at click time.
- `BallTracker` rejects NaN and >30m-from-court-centre positions.
- Player position log drops (0,0) failed projections and out-of-court noise.

Branch and its worktree deleted after porting.

## Ops / infrastructure

- Root cause of the "Mac runs hot" report: a 16-hour orphaned analysis of a
  3.3 GB video at 7% progress (closing the browser does not stop a server-side
  job). Backend now cancels active analyses on shutdown (regression-tested —
  previously a plain SIGTERM hung and needed kill -9) and logs progress with
  ETA every 1%.
- Repo consolidated to a single `master` branch locally; stale remote branches
  listed for manual deletion (permission-gated).
- Disk cleanup: duplicate 3.3 GB uploaded video removed, unused Ollama models
  (moondream, qwen2.5vl:3b, qwen3-vl:2b thinking) removed — 10→18 GB free.
- README rewritten: plain-English overview, dated journey log, updated
  architecture diagram (OSNet, VLM audit loop).

## Next actions (priority order)

1. **Download PadelTracker100** (paused at user request; partial 360 MB kept,
   `curl -C -` resumes). Then: `scripts/convert_padeltracker100.py` →
   `scripts/merge_ball_labels.py` → fine-tune via
   `scripts/train_temporal_ball.py` (freeze-backbone first on MPS) → must beat
   the 63.5% gate on `eval_ball_labels.py --split test` before adoption.
2. **Re-run full match analysis** with new ball model + OSNet re-ID; measure
   scoring accuracy movement end-to-end.
3. **Run `scripts/eval_players.py`** against PADELVIC player GT to quantify
   the OSNet gain and tune re-ID thresholds (current: 0.60 sim / 0.06 margin).
4. **Delete stale remote branches** (manual, permission-gated):
   `git push origin --delete feat/eval-harnesses-3d-z feature/accuracy-fixes
   main phase2-cv-pipeline phase3-frontend-rebuild tracknetv2-integration`
5. Parked ideas (see README Journey): pseudo-labeling loop, Grounding DINO
   auto-labeler, T-DEED event spotting on PadelTracker100 shot labels,
   Deep-EIoU association, TrackNetV4 motion-attention retrofit.
