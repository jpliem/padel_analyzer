# Standalone VLM Match Coach

> **Current status:** local inference works, but automatic rally scoring is
> still research. See
> [current-vlm-scoring-status.md](current-vlm-scoring-status.md) for tested
> timings, known limitations, and the recommended hybrid direction.

This is a separate, local-first application under `vlm_coach/`. It does not run
the existing frame-by-frame referee pipeline.

Its product promise is:

> Turn a long padel recording into evidence-linked rally notes, highlights, a
> match story, and practical training priorities.

## How it works

1. The user uploads a fixed-camera match recording.
2. Lightweight motion sampling finds candidate active-play windows.
3. The app extracts eight chronological, resized JPEGs from every window.
4. Local Qwen receives one storyboard at a time through Apple MLX (preferred)
   or Ollama.
5. Every response is validated against a strict rally-analysis JSON schema.
6. A second text-only pass combines the rally JSON into a match story.
7. The user can correct the likely winner, ending, and coaching note for any
   segment, then rebuild the story from the reviewed evidence.

The app deliberately asks Qwen to mark unsupported winners, bounces, glass
contacts, and line calls as unknown. It is a coaching-story system, not an
automatic referee.

## Hardware profile

The default `qwen3.5:2b` model is intended for the repository owner's 8 GB Apple
M2. Storyboards are processed sequentially, images are capped at 768 pixels
wide, the context is capped at 8192 tokens, extended thinking is disabled, and
responses are short JSON documents.

`qwen3.5:0.8b` is available as a faster, lower-quality option. `qwen3.5:4b` may
run on 8 GB but leaves less room for the browser, video decoding, and context.

### Measured on the target M2 / 8 GB Mac

Using the local 4-bit `qwen3.5:0.8b` MLX model and the repository's real padel
footage:

- four images from a 10-second window: 12.605 seconds (1.261x real-time);
- eight images from a 10-second window: 13.659 seconds (1.366x real-time);
- full app flow on a 9.968-second clip, including the text-only story pass:
  14.175 seconds of VLM work (1.422x real-time).

This proves the Mac can run the app locally. The 0.8B model is fast but its
coaching interpretation can be shallow. The app therefore defaults to 2B for
quality and refuses to invent advice when the rally JSON contains no tactical
evidence. Use 0.8B for quick previews and 2B for the intended review mode.

## Setup and run

The app reuses the backend Python environment. On Apple Silicon, install the
MLX provider and run:

```bash
backend/.venv/bin/pip install -r backend/requirements-vlm-coach.txt
bash run_vlm_coach.sh
```

Open <http://127.0.0.1:8765>.

The launcher prefers in-process MLX on Apple Silicon. The selected 4-bit Qwen
model is downloaded from Hugging Face on first use and then cached locally.
The launcher defaults to standard resumable HTTPS because the Hugging Face Xet
transport was unreliable on the target network.
If MLX is unavailable, the launcher falls back to Ollama and starts its daemon
when needed. Override the provider with `VLM_COACH_PROVIDER=mlx|ollama`, the
port with `VLM_COACH_PORT`, and the Ollama address with `OLLAMA_URL`.

Match data is stored separately under `vlm_coach/data/matches/` and ignored by
Git. Each match directory contains the uploaded recording, persistent JSON, and
storyboard images.

## API outline

- `GET /api/health` — service, provider, and available-model status.
- `POST /api/matches` — upload a recording and create a match.
- `GET /api/matches` — list locally stored reviews.
- `GET /api/matches/{id}` — retrieve progress, rallies, and story.
- `POST /api/matches/{id}/analyze` — start local analysis.
- `PATCH /api/matches/{id}/rallies/{rally}` — save human evidence.
- `POST /api/matches/{id}/story` — rebuild the story after corrections.

## Verification

```bash
cd backend
.venv/bin/python -m pytest -q
```

The standalone tests are `backend/tests/test_vlm_coach.py`.

Run a real one-window benchmark with:

```bash
backend/.venv/bin/python -m vlm_coach.benchmark \
  --provider mlx --model qwen3.5:0.8b \
  --video data/test_footage/padel_test.mp4 --start 0 --duration 10
```
