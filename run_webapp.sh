#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$ROOT/backend/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Backend environment is missing. Run: cd backend && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Frontend packages are missing. Run: cd frontend && npm install"
  exit 1
fi

cleanup() {
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

API_PORT="${API_PORT:-8000}"
while lsof -nP -iTCP:"$API_PORT" -sTCP:LISTEN >/dev/null 2>&1; do
  API_PORT=$((API_PORT + 1))
done

echo "Starting Padel Smart Recording"
echo "Web app: http://localhost:3000"
echo "API:     http://localhost:$API_PORT"

(
  cd "$ROOT/backend"
  PORT="$API_PORT" exec "$PYTHON" main.py
) &
BACKEND_PID=$!

cd "$ROOT/frontend"
FAST_REFRESH=true REACT_APP_API_PORT="$API_PORT" npm start
