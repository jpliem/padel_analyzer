# Padel Analyzer

Computer-vision match analyzer for padel. Upload a fixed-camera recording of a match; the system tracks the ball and players, detects events (serves, bounces, wall contacts, rallies), proposes point decisions under real padel rules, and presents everything in a web dashboard with an auditable review queue and a 2D/3D court view.

**Design principle: the system never invents certainty.** Computer vision on a single camera is imperfect, so every automatic point decision goes into a review ledger for human confirmation. The displayed score only includes confirmed points.

## What is this? (plain English)

You film a padel match with one fixed camera and feed the video to this app. It:

1. **Finds the ball** in every frame (a tiny, fast, motion-blurred dot — the hardest part).
2. **Follows the four players** and keeps their identities straight, even after they cross or leave frame.
3. **Hears racket contacts** in the audio and combines them with ball direction changes.
4. **Understands the game** — serves, bounces, wall hits, rally start/end — and applies real padel (FIP) rules to propose "team A won this point, because X".
5. **Shows you everything** on a web dashboard: score proposals, rally replays, a 3D court view of the ball's flight.

The twist versus other "AI sports analyzers": **it never pretends to be sure.** Every proposed point lands in a review queue with its evidence; a human confirms or rejects it. The scoreboard only ever shows human-confirmed points. All accuracy numbers below are measured on hand-labeled frames, not marketing.

## Quick start

### Backend (Python 3.12, FastAPI)

```bash
cd backend
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py        # API on http://localhost:8000
```

### Frontend (React + TypeScript)

```bash
cd frontend
npm install
npm start                       # Dev server on http://localhost:3000
```

### Workflow

1. **Match setup** — players, teams, format, golden point.
2. **Calibration** — click court corners (or 12 keypoints) on a video frame; optional net posts for 3D.
3. **Upload & analyze** — background job with progress; results persist across restarts.
4. **Review** — score proposals, rally highlights, event log, 2D/3D court replay. Confirm or reject each proposed point.

## Architecture

```
video ─▶ ball detector (TrackNet) ─▶ active-ball gate ─▶ Kalman track
                                                        │
        players (YOLO + ReID) ──────────────────────────┤
        audio impulses ─────────────────────────────────┤
                                                        ▼
        monocular ballistic fit (gravity-constrained z) │
                                                        ▼
        event detectors ─▶ semantic bridge ─▶ padel rules engine
                                                        ▼
        review ledger (human-gated scoring) ─▶ API ─▶ React dashboard
```

- `backend/src/cv/` — detectors (pluggable via `detector_type`: `tracknet`, `yolo`, `fast`), trackers, calibration, monocular 3D trajectory fitting.
- `backend/src/logic/` — event detection, match state machine, FIP rules engine, contact fusion, review ledger.
- `backend/src/pipeline/VideoAnalyzer` — orchestrates everything; also live mode via WebSocket.
- `frontend/src/pages/` — MatchSetup → Calibration → OfflineAnalysis / LiveView.

## Ball height from one camera

A single camera cannot geometrically measure ball height. The analyzer fits a gravity-constrained ballistic segment (`p(t) = p0 + v0·t + ½g·t²`) to windows of pixel rays and only trusts the fitted z when the fit passes reliability gates (ray error, conditioning, positive depths). Unreliable windows keep z at the ground estimate — visible in the 3D court view as a flat trail.

## Accuracy (measured, not marketed)

Evaluated on reviewed, hand-labeled real-footage frames (`scripts/eval_ball_labels.py` is the authoritative gate; labels in `data/labels/padelvic_panasonic_combined/`):

| Detector | Precision/Recall @15px (held-out test) |
|---|---|
| `tracknet_padel.pt` conf 0.3 (production) | 63.5% / 63.5% |
| `tracknet_padel.pt` conf 0.5 | 70.0% / 53.9% |

The model registry (`backend/src/cv/model_registry.py`) records the production model's benchmark evidence and limitations; the dashboard displays them. Player detection, event detection, and scoring inherit these limits — hence the human review gate.

## Tests

```bash
cd backend && .venv/bin/python -m pytest      # 550+ tests
cd frontend && npm run build                  # type-check + production build
```

Detector changes must be validated against the label gate before merging:

```bash
backend/.venv/bin/python scripts/eval_ball_labels.py \
  --labels data/labels/padelvic_panasonic_combined/labels.json \
  --detector tracknet --tracknet-model models/tracknet_padel.pt \
  --split test --threshold-px 15
```

## VLM track audit (optional)

A local vision-language model audits the tracker's output after analysis. Since 2026-07-19 the audit uses **pointing, not judgment**: the model is shown a full-resolution crop around the tracker's claimed position and asked to point at the ball; the auditor computes the pointed-vs-claimed distance in code and derives the verdict. Disagreement frames become the next labeling queue (active learning). Verdicts never change the score.

```bash
ollama pull qwen3-vl:2b-instruct   # once (use -instruct tags: thinking variants return empty JSON)
backend/.venv/bin/python -m vlm_coach.track_audit \
  --results /path/to/results.json \
  --video /path/to/video.mp4 --calib /path/to/calib.json \
  --sample 12 --out /tmp/track_audit.json
```

**Why pointing:** measured on labeled frames (`scripts/eval_vlm_auditor.py`), every small local VLM answers "is the marker on the ball" with a fixed bias — qwen3-vl:2b-instruct says yes to everything, qwen2.5vl:3b says no to everything, moondream says unclear to everything — on full frames *and* crops. But qwen3-vl:2b-instruct can *point* at the ball on full-res crops with ≤10px error (4/6 found; coordinates arrive 0-1000 normalized). So the model locates, the code judges. `--mode judgment` keeps the legacy behaviour for comparison.

**Memory requirement:** ~2 GB for the 2B model. On an 8 GB machine, run the audit while the analyzer and dev servers are stopped.

## Production notes

- **Deployment scope:** designed for self-hosted, single-operator use on a trusted LAN. There is no authentication layer; do not expose the API to the public internet without adding one (reverse proxy + auth).
- Upload size capped via `PADEL_MAX_UPLOAD_BYTES` (default 8 GB). Path identifiers validated server-side. Analysis errors are logged server-side; clients get generic messages.
- Analysis runs as a background task in-process. For multi-user scale, move `analyze` to a worker queue. A running analysis can be cancelled from the UI (or `POST /analyze/cancel/{job_id}`); it stops within one frame.
- Full-resolution TrackNet on CPU is heavy (~2-7 fps). Expect a 90-minute match to take hours and keep the machine warm; run analyses one at a time.
- Match data persists as JSON under `backend/data/matches/{match_id}/`.

## Troubleshooting

- **Port 8000 busy** — run the API elsewhere and point the frontend at it:
  `uvicorn main:app --port 8001` + `REACT_APP_API_PORT=8001 npm start`.
- **Machine hot / analysis slow** — expected: full-resolution TrackNet on CPU runs at 2-7 fps, so a full match takes hours. Use the **Cancel analysis** button (or `POST /analyze/cancel/{job_id}`) to stop within one frame; the job is marked cancelled and can be retried.
- **No ball height in results** — height needs 3D calibration. Calibrate with 12 keypoints (or net-top points); a 4-corner homography alone cannot produce z, and the 3D court view will show a flat trail.
- **Detector changes look better but score worse** — trust only the label gate (`scripts/eval_ball_labels.py --split test`), never a vibe check on one clip.

## Journey

Dated log of what we tried, what we measured, and what it taught us.

### 2026-03-25 → 03-26 — Core build
Full pipeline assembled: court model with 3D wall geometry, wall-collision detection, multi-camera fusion scaffolding (`WorldFusion`, `CameraNode`), match API, live mode, React dashboard. End-to-end integration tests.

### 2026-06-17 → 06-19 — The 3D push
- Built real two-camera triangulation (DLT + reprojection gating + audio/motion camera sync). **Result: it works** — real ball heights (median 1.78 m) vs single-camera z≡0.
- **Decision:** multi-camera stays a *validation harness*, not the product — consumers film with one phone. Product path: single camera + gravity-constrained ballistic fit for z (`docs/multicam-3d-progress.md`).
- Colour+motion ball candidates + epipolar matching cracked the "tracker locks onto a player's head" failure.
- **Key finding:** single-camera ball positions were ~1/3 physically impossible (off-court, 1000 km/h) — ball detection quality is the keystone blocking serve detection, rally detection, and scoring.

### 2026-07-17 → 07-18 — Honesty pass + real baselines
- Hand-labeled real-footage ball frames (197 reviewed labels, leak-safe splits). **Measured production detector: 63.5% P/R @15px** — the honest number behind everything else.
- Discovered PADELVIC synthetic data is out-of-domain: a model scoring well there collapsed on real footage (`tracknet_phase1.pt`, 0% — never use).
- Built the honest-scoring stack: FIP rules engine over semantic observations, review ledger (CV can never auto-confirm a point), visibility state machine (a detector miss is uncertainty, never an OUT call), audio contact fusion as evidence-only.
- Production hardening: cancellation, persistence, security on IDs, upload caps (`docs/changelog-2026-07-18-production-hardening.md`).

### 2026-07-19 — Research day + auditor redesign
- **SOTA research sweep** (verified sources): TrackNetV3/V4-class models hit 94-98% on tennis/shuttlecock vs our 63.5% — retraining is the biggest available win. Deep-EIoU/HM-SORT beat ByteTrack-style tracking in sports (HOTA 0.42→0.54 from association change alone). Motion-blur-aware labeling (BlurBall) cuts trajectory error ~40%.
- **Found the missing dataset:** PadelTracker100 (Zenodo 14653706, CC-BY-4.0) — ~100k annotated real padel frames from WPT Finals 2022 with ball, player, pose, and shot-event labels. Converter written (`scripts/convert_padeltracker100.py`).
- **VLM auditor redesigned after measurement:** small local VLMs cannot *judge* a marker (fixed-bias answers) but qwen3-vl:2b-instruct can *point* at the ball within ≤10px on full-res crops. Audit now = crop → point → distance in code. First live run correctly exposed the tracker pointing at a sideline barrier — the auditor now catches exactly the noise the 06-19 finding predicted.
- **Player re-ID upgraded:** HSV histograms (measured: cannot separate similar kits at all, −0.003) replaced by OSNet x0.25 embeddings (+0.042 separation on the same crops; vendored, MPS, 8.9 MB).
- **Hardware reality documented:** dev machine is M2/8 GB — Molmo 2, Qwen3.5-VL, SAM 3 don't fit; qwen3-vl:2b-instruct is the working local VLM. Big-model experiments go to cloud batch.
- Ops: fixed a stale second Ollama server silently breaking model loads; found a 16-hour orphaned analysis job cooking the machine (3.3 GB video at 7%); backend now cancels analyses on shutdown and logs progress + ETA.

### Ideas parked for later
- Pseudo-labeling loop: fine-tune detectors on their own high-confidence predictions (measured elsewhere: HOTA 0.380→0.491, false positives 4913→494).
- Grounding DINO as zero-shot auto-labeler feeding the human review queue (beats YOLO-World on small objects; too slow for production inference).
- T-DEED-style precise event spotting head trained on PadelTracker100's shot labels (replaces heuristic serve/hit detection).
- Deep-EIoU / harmonic-mean association to replace ByteTrack-style matching (keeps all tracklets — padel is a closed 4-player world).
- TrackNetV4 motion-attention layer retrofit; blur-aware ball labels.

### Next up (as of 2026-07-19)
1. Download PadelTracker100 → convert → retrain ball model → must beat the 63.5% gate.
2. Re-run full match analysis with new ball model + OSNet re-ID; measure scoring accuracy movement.

## Docs

- `docs/changelog-2026-07-18-production-hardening.md` — hardening session: bugs found/fixed, measured baselines, verification runs, remaining gaps.
- `docs/multicam-3d-progress.md` — why the architecture is single-camera physics fitting (multi-camera triangulation is the validation harness, not the product).
- `CLAUDE.md` — full module map and repo conventions.

## Datasets & research scripts

`scripts/` contains the evaluation and research harnesses (multi-camera triangulation, camera sync, label tooling, model benchmarks). These are research tools, not part of the production path; `docs/multicam-3d-progress.md` records the findings that shaped the current architecture.
