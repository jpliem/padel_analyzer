# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Padel match analyzer using computer vision: tracks ball and players, detects events (bounces, serves, hits), auto-scores using padel rules, and provides a web dashboard with 3D court visualization.

## Commands

### Backend (from `backend/`)
```bash
# Run API server
python main.py                          # Starts FastAPI on port 8000

# Run all tests
python -m pytest

# Run a single test file
python -m pytest tests/test_ball_tracker.py

# Run a specific test
python -m pytest tests/test_scoring_engine.py::test_golden_point -v

# Install dependencies (venv lives at backend/.venv/)
.venv/bin/pip install -r requirements.txt
```

### Frontend (from `frontend/`)
```bash
npm start    # Dev server on port 3000
npm run build
```

## Architecture

### Backend (`backend/src/`)

**Pipeline layer** (`pipeline/`): Orchestrates everything.
- `VideoAnalyzer` is the central class — wires together detectors, trackers, event detection, and scoring. Accepts `detector_type` param: `"yolo"`, `"tracknet"`, or `"fast"`.
- `LiveManager` wraps VideoAnalyzer for real-time camera/RTSP feeds with WebSocket streaming.
- `ReplayBuffer` stores recent frames for instant replay.

**CV layer** (`cv/`): Computer vision components.
- `detectors/` — pluggable ball detectors sharing a `BaseBallDetector` interface (`base.py`): `YoloBallDetector`, `TrackNetBallDetector`, `FastBallDetector` (frame differencing). `UnifiedYoloDetector` runs a single YOLO model shared between ball and player detection.
- `BallTracker` — Kalman filter smoothing, trajectory history, 3D height estimation (parabola fitting + pixel velocity). Falls back to pixel-space coordinates when uncalibrated.
- `active_ball.py` — temporal gating that picks the in-play ball from detector candidates and rejects implausible jumps.
- `monocular_trajectory.py` — gravity-constrained ballistic fit over a rolling window of pixel rays; gives real single-camera ball height (z) when the fit passes reliability gates (ray error, conditioning, positive depth, playable-volume bounds). Requires a `CameraModel` (pixel_ray); plain homography calibration cannot feed it.
- `visibility.py` — explicit ball visibility state machine; a detector miss is uncertainty, never an OUT call.
- `audio_events.py` — audio impulse detection (racket contacts) via ffmpeg.
- `PlayerTracker` — YOLO + ByteTrack-style ID persistence, team assignment; `player_reid.py` HSV-histogram gallery reconnects IDs after track loss.
- `CourtCalibration` — 2D homography from 4-corner or 12-keypoint calibration. `CameraModel` — full 3D camera calibration with PnP solve; enables `pixel_ray` and the monocular fit.
- `CourtDetector` — automatic court keypoint detection from a video frame.
- `model_registry.py` — declares the production ball model plus its measured benchmark evidence and limitations (surfaced in the UI).

**Logic layer** (`logic/`): Game understanding.
- `EventDetector` aggregates domain-specific detectors from `logic/detectors/`: `BounceDetector`, `ServeDetector`, `LastHitterDetector`, `PointEndDetector`.
- `MatchStateMachine` — tracks match state transitions (IDLE → SERVING → RALLY → POINT_ENDED).
- `PadelScoringEngine` — padel scoring rules (15/30/40/game, deuce/advantage, golden point, sets, tiebreaks).
- `padel_rules.py` — deterministic FIP rules engine over semantic `PadelObservation`s; low-confidence observations become REVIEW_REQUIRED, visibility loss never ends a point.
- `semantic_bridge.py` — translates legacy CV `MatchEvent`s into `PadelObservation`s.
- `contact_fusion.py` — fuses audio + direction-change + proximity into contact proposals (diagnostic/review material only; never scores).
- `review_ledger.py` — auditable point ledger. **Core honesty rule: CV proposals use `auto_confirm_threshold=2.0`, so CV never auto-confirms a point — the displayed score contains only human-confirmed points.**

**Models** (`models/`): Shared data types (`types.py`) and configuration (`config.py`). Key enums: `EventType`, `MatchState`, `TeamId`, `PointReason`.

### API (`backend/main.py`)

Single FastAPI app. Match lifecycle: setup → calibrate → upload video → analyze. Analysis runs as a background task with progress polling. Also supports live mode via WebSocket at `/live/stream`. Match data persisted as JSON in `data/matches/{match_id}/`.

### Frontend (`frontend/src/`)

React + TypeScript. Pages follow the match workflow: `MatchSetup` → `Calibration` (with `CalibrationCanvas` for point picking) → `OfflineAnalysis` / `LiveView`. 3D court visualization uses Three.js via `@react-three/fiber` in `Court3DView`. API client in `api.ts`.

### VLM coach & audit (`vlm_coach/`, top level)

Separate module (not under `backend/src`). Local VLM (Ollama/MLX) layers on top of the CV pipeline:
- `pipeline.py` — storyboard-based rally coaching reports (honest prompts; never invents scores).
- `track_audit.py` — renders tracker claims onto frames, VLM judges "is the marker on the actual ball"; disagreements become the labeling queue. Verdicts never change the score.
- Tests live in `backend/tests/test_vlm_coach.py` / `test_track_audit.py` (repo root on sys.path via fixtures).

## Key Patterns

- **pytest config**: `pythonpath = src` in `pytest.ini` — imports use `from cv.ball_tracker import ...` (no `src.` prefix).
- **Detector plugin pattern**: All ball detectors implement `BaseBallDetector.detect(frame) -> list[Detection]`. Swap via `detector_type` parameter.
- **Calibration dual-path**: `CourtCalibration` (2D homography) always computed; `CameraModel` (3D PnP) computed when enough data available. `_load_calibration()` in `main.py` picks the best available.
- **YOLO model**: `yolov8n.pt` lives in `backend/` root. `UnifiedYoloDetector` runs it once per frame for both ball and player detections.
- **Detector accuracy gate**: any ball-detector change must be validated with `scripts/eval_ball_labels.py --labels data/labels/padelvic_panasonic_combined/labels.json --split test` before adoption. Current production: `tracknet_padel.pt` conf 0.3 → 63.5% P/R @15px. `tracknet_phase1.pt` is a collapsed retrain (0%) — never use it.
- **Honest-scoring policy**: evidence (CV events, audio, contact fusion, VLM audit verdicts) creates review material; it never awards a point by itself. Do not bypass the review ledger.
- **Identifier safety**: all match/template IDs pass `_require_safe_id()` in `main.py` before touching the filesystem.
- **Session history**: `docs/changelog-2026-07-18-production-hardening.md` records the hardening pass, measured baselines, and remaining gaps. `docs/multicam-3d-progress.md` records why the architecture is single-camera + physics (multicam is the validation harness, not the product).
