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

# Install dependencies (uses venv at backend/venv/)
pip install -r requirements.txt
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
- `BallTracker` — Kalman filter smoothing, trajectory history, 3D height estimation (parabola fitting + pixel velocity).
- `PlayerTracker` — YOLO + ByteTrack-style ID persistence, team assignment, player-to-court-position mapping.
- `CourtCalibration` — 2D homography from 4-corner or 12-keypoint calibration. `CameraModel` — full 3D camera calibration with PnP solve for height estimation.
- `CourtDetector` — automatic court keypoint detection from a video frame.

**Logic layer** (`logic/`): Game understanding.
- `EventDetector` aggregates domain-specific detectors from `logic/detectors/`: `BounceDetector`, `ServeDetector`, `LastHitterDetector`, `PointEndDetector`.
- `MatchStateMachine` — tracks match state transitions (IDLE → SERVING → RALLY → POINT_ENDED).
- `PadelScoringEngine` — padel scoring rules (15/30/40/game, deuce/advantage, golden point, sets, tiebreaks).

**Models** (`models/`): Shared data types (`types.py`) and configuration (`config.py`). Key enums: `EventType`, `MatchState`, `TeamId`, `PointReason`.

### API (`backend/main.py`)

Single FastAPI app. Match lifecycle: setup → calibrate → upload video → analyze. Analysis runs as a background task with progress polling. Also supports live mode via WebSocket at `/live/stream`. Match data persisted as JSON in `data/matches/{match_id}/`.

### Frontend (`frontend/src/`)

React + TypeScript. Pages follow the match workflow: `MatchSetup` → `Calibration` (with `CalibrationCanvas` for point picking) → `OfflineAnalysis` / `LiveView`. 3D court visualization uses Three.js via `@react-three/fiber` in `Court3DView`. API client in `api.ts`.

## Key Patterns

- **pytest config**: `pythonpath = src` in `pytest.ini` — imports use `from cv.ball_tracker import ...` (no `src.` prefix).
- **Detector plugin pattern**: All ball detectors implement `BaseBallDetector.detect(frame) -> list[Detection]`. Swap via `detector_type` parameter.
- **Calibration dual-path**: `CourtCalibration` (2D homography) always computed; `CameraModel` (3D PnP) computed when enough data available. `_load_calibration()` in `main.py` picks the best available.
- **YOLO model**: `yolov8n.pt` lives in `backend/` root. `UnifiedYoloDetector` runs it once per frame for both ball and player detections.
