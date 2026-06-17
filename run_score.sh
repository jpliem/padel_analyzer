#!/usr/bin/env bash
#
# Run from anywhere:
#     bash /Users/jonathan/Documents/Github/padel_analyzer/run_score.sh
#
# 1. annotate the triangulated 3D ball (with height) onto the panasonic video
# 2. feed the gated 3D into the existing scoring brain -> events + score
# 3. grade the rally detection vs the Plays-sheet ground truth
#
# Uses /tmp/tri_gated.json (from the gated triangulation run). If missing,
# run run_debug.sh first.

set -uo pipefail
ROOT="/Users/jonathan/Documents/Github/padel_analyzer"
PY="$ROOT/venv/bin/python"; [ -x "$PY" ] || PY="$ROOT/backend/venv/bin/python"
cd "$ROOT"
XLSX="data/datasets/padelvic/derived/PadelVic_Panasonic_labeling.xlsx"

if [ ! -f /tmp/tri_gated.json ]; then
  echo "missing /tmp/tri_gated.json — run run_debug.sh first"; exit 1
fi

echo "════════════════════════════════════════"
echo " 1) ANNOTATE TRIANGULATED 3D BALL ON VIDEO"
echo "════════════════════════════════════════"
"$PY" scripts/annotate_3d.py --points /tmp/tri_gated.json \
  --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
  --calib /tmp/panasonic_cammodel2.json --start 2 --window 20 \
  --out /tmp/ball_3d.mp4 2>&1 | grep -vE "Warning|warn"

echo
echo "════════════════════════════════════════"
echo " 2) FEED 3D INTO SCORING BRAIN"
echo "════════════════════════════════════════"
"$PY" scripts/score_from_3d.py --points /tmp/tri_gated.json \
  --first-server near --out /tmp/score_3d.json 2>&1 | grep -vE "Warning|warn"

echo
echo "════════════════════════════════════════"
echo " 3) GRADE RALLY DETECTION vs GROUND TRUTH"
echo "════════════════════════════════════════"
MAXF=$("$PY" -c "import json;print(json.load(open('/tmp/score_3d.json'))['frames_processed'])")
"$PY" scripts/eval_rallies.py --results /tmp/score_3d.json --xlsx "$XLSX" \
  --max-frame "$MAXF" 2>&1 | grep -vE "Warning|warn"

echo
echo "════════════════════════════════════════"
echo " OUTPUTS"
echo "════════════════════════════════════════"
echo "  /tmp/ball_3d.mp4    annotated 3D ball video  (open /tmp/ball_3d.mp4)"
echo "  /tmp/score_3d.json  events + score from triangulated 3D"
echo "DONE."
