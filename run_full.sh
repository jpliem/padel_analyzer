#!/usr/bin/env bash
#
# Full multi-camera pipeline, fresh, with the ball-vs-head filter:
#     bash /Users/jonathan/Documents/Github/padel_analyzer/run_full.sh
#
# 1. triangulate the ball from panasonic+gopro (reproj-gated + head-filtered)
# 2. annotate the triangulated 3D ball (height) onto the video
# 3. feed gated 3D into the scoring brain
# 4. grade rally detection vs ground truth
#
# ~8-12 min (player detection runs per frame on both cameras now).

set -uo pipefail
ROOT="/Users/jonathan/Documents/Github/padel_analyzer"
PY="$ROOT/venv/bin/python"; [ -x "$PY" ] || PY="$ROOT/backend/venv/bin/python"
cd "$ROOT"
XLSX="data/datasets/padelvic/derived/PadelVic_Panasonic_labeling.xlsx"

echo "═══ 1) TRIANGULATE (reproj-gated + head-filtered) ~6-10min ═══"
"$PY" scripts/triangulate_ball.py \
  --video-a data/datasets/padelvic/cameras/panasonic_final.mp4 --calib-a /tmp/panasonic_cammodel2.json \
  --video-b data/datasets/padelvic/cameras/gopro.mp4 --auto-court-b \
  --sync /tmp/sync_pana_gopro.json --start 2 --window 40 --rate 10 \
  --max-reproj 20 --out /tmp/tri_gated.json 2>&1 | grep -vE "Warning|warn"

echo; echo "═══ 2) ANNOTATE 3D BALL ON VIDEO ═══"
"$PY" scripts/annotate_3d.py --points /tmp/tri_gated.json \
  --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
  --calib /tmp/panasonic_cammodel2.json --start 2 --window 20 \
  --out /tmp/ball_3d.mp4 2>&1 | grep -vE "Warning|warn"

echo; echo "═══ 3) SCORE FROM 3D ═══"
"$PY" scripts/score_from_3d.py --points /tmp/tri_gated.json \
  --first-server near --out /tmp/score_3d.json 2>&1 | grep -vE "Warning|warn"

echo; echo "═══ 4) GRADE vs GROUND TRUTH ═══"
MAXF=$("$PY" -c "import json;print(json.load(open('/tmp/score_3d.json'))['frames_processed'])")
"$PY" scripts/eval_rallies.py --results /tmp/score_3d.json --xlsx "$XLSX" --max-frame "$MAXF" 2>&1 | grep -vE "Warning|warn"

echo; echo "OUTPUTS: /tmp/ball_3d.mp4  /tmp/tri_gated.json  /tmp/score_3d.json"
echo "DONE."
