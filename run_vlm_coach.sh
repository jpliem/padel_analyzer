#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/backend/.venv/bin/python"
PORT="${VLM_COACH_PORT:-8765}"
PROVIDER="${VLM_COACH_PROVIDER:-auto}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment missing. Run: cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
HAS_MLX="$($PYTHON -c 'import importlib.util; print("yes" if importlib.util.find_spec("mlx_vlm") else "no")')"
if [[ "$PROVIDER" == "mlx" || ( "$PROVIDER" == "auto" && "$HAS_MLX" == "yes" ) ]]; then
  echo "VLM runtime: MLX (Apple Silicon)"
elif [[ "$PROVIDER" == "ollama" || "$PROVIDER" == "auto" ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Neither mlx-vlm nor Ollama is available."
    echo "Install the standalone dependencies: backend/.venv/bin/pip install -r backend/requirements-vlm-coach.txt"
    exit 1
  fi
  if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "Starting Ollama…"
    ollama serve >"${TMPDIR:-/tmp}/padel-vlm-coach-ollama.log" 2>&1 &
    OLLAMA_PID=$!
    trap 'kill "$OLLAMA_PID" 2>/dev/null || true' EXIT INT TERM
    for _ in {1..20}; do
      curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
      sleep 0.25
    done
  fi
  echo "VLM runtime: Ollama"
else
  echo "VLM_COACH_PROVIDER must be auto, mlx, or ollama"
  exit 1
fi

echo "Padel Match Coach: http://127.0.0.1:$PORT"
cd "$ROOT"
VLM_COACH_PORT="$PORT" exec "$PYTHON" -m vlm_coach.app
