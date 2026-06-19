# Repository Guidelines

## Project Structure & Module Organization

This repository contains a padel match analyzer with a Python computer-vision backend and a React/TypeScript frontend. Backend entry points are `backend/main.py` and `backend/cli_analyze.py`. Core backend modules are under `backend/src/`: `cv/` for tracking, calibration, and triangulation; `logic/` for scoring and event detection; `models/` for shared types; and `pipeline/` for orchestration. Backend tests live in `backend/tests/`.

Frontend code is in `frontend/src/`, with pages in `pages/`, UI in `components/`, API access in `api.ts`, hooks in `hooks/`, and shared styles in `styles/global.css`. Utility pipelines live in `scripts/`; notes are in `docs/`; video data is under `data/`.

## Build, Test, and Development Commands

Run backend commands from `backend/`:

```bash
pip install -r requirements.txt
python main.py
python -m pytest
python -m pytest tests/test_scoring_engine.py::test_golden_point -v
```

`python main.py` starts the FastAPI server on port 8000. `python -m pytest` runs the backend test suite using `backend/pytest.ini`.

Run frontend commands from `frontend/`:

```bash
npm install
npm start
npm run build
```

`npm start` runs the development server on port 3000. `npm run build` creates the production bundle. Root scripts such as `bash run_full.sh`, `bash run_debug.sh`, and `bash run_score.sh` run longer video-analysis pipelines and may expect local dataset files.

## Coding Style & Naming Conventions

Use Python modules and tests with `snake_case` names. Keep imports compatible with `pythonpath = src`, for example `from cv.ball_tracker import BallTracker`, not `from src.cv...`. Prefer shared model types from `backend/src/models/` for cross-module data.

Use React components with `PascalCase` filenames and exports, hooks as `useSomething`, and shared TypeScript types in `frontend/src/types.ts`. Keep CSS class names descriptive.

## Testing Guidelines

Backend tests use `pytest`; name files `test_*.py` and place them in `backend/tests/`. Add focused tests for scoring rules, event detection, calibration, and pipeline behavior. For frontend changes, run `npm run build` because no frontend test script is currently defined.

## Commit & Pull Request Guidelines

Recent commits use concise Conventional Commit-style prefixes such as `feat:` and `docs:`. Follow that pattern, for example `feat: add rally triangulation diagnostics` or `fix: handle missing calibration points`.

Pull requests should summarize behavior changes, list verification commands, call out dataset/model assumptions, and include screenshots or clips for UI or visualization changes. Link related issues or docs when available.

## Security & Configuration Tips

Do not commit local `.env` files, generated caches, virtual environments, or large private datasets. Treat model weights and videos as local artifacts unless intentionally versioned.
