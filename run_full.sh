#!/usr/bin/env bash
#
# Full multi-camera pipeline with the FIXED gopro calibration:
#     bash /Users/jonathan/Documents/Github/padel_analyzer/run_full.sh
#
# 1. triangulate panasonic + gopro (both manually calibrated, reproj-gated + head-filtered)
# 2. 3D reconstruction view (court + ball)
# 3. annotate the triangulated 3D ball onto the video
# 4. feed 3D into scoring + grade vs ground truth
#
# ~8-12 min.

set -uo pipefail
ROOT="/Users/jonathan/Documents/Github/padel_analyzer"
PY="$ROOT/venv/bin/python"; [ -x "$PY" ] || PY="$ROOT/backend/venv/bin/python"
cd "$ROOT"
XLSX="data/datasets/padelvic/derived/PadelVic_Panasonic_labeling.xlsx"

echo "═══ 1) TRIANGULATE (panasonic + FIXED gopro calib) ~6-10min ═══"
"$PY" scripts/triangulate_ball.py \
  --video-a data/datasets/padelvic/cameras/panasonic_final.mp4 --calib-a /tmp/panasonic_cammodel2.json \
  --video-b data/datasets/padelvic/cameras/gopro.mp4 --calib-b /tmp/gopro_cammodel.json \
  --sync /tmp/sync_pana_gopro.json --start 2 --window 40 --rate 10 \
  --max-reproj 20 --out /tmp/tri_gated.json 2>&1 | grep -vE "Warning|warn"

echo; echo "═══ 2) 3D RECONSTRUCTION VIEW ═══"
"$PY" scripts/view_3d.py --points /tmp/tri_gated.json --out /tmp/court_3d 2>&1 | grep -vE "Warning|warn"

echo; echo "═══ 3) ANNOTATE 3D BALL ON VIDEO ═══"
"$PY" scripts/annotate_3d.py --points /tmp/tri_gated.json \
  --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
  --calib /tmp/panasonic_cammodel2.json --start 2 --window 20 \
  --out /tmp/ball_3d.mp4 2>&1 | grep -vE "Warning|warn"

echo; echo "═══ 4) SCORE FROM 3D + GRADE ═══"
"$PY" scripts/score_from_3d.py --points /tmp/tri_gated.json --first-server near --out /tmp/score_3d.json 2>&1 | grep -vE "Warning|warn"
MAXF=$("$PY" -c "import json;print(json.load(open('/tmp/score_3d.json'))['frames_processed'])")
"$PY" scripts/eval_rallies.py --results /tmp/score_3d.json --xlsx "$XLSX" --max-frame "$MAXF" 2>&1 | grep -vE "Warning|warn"

echo; echo "OUTPUTS: /tmp/court_3d_*.png  /tmp/ball_3d.mp4  /tmp/tri_gated.json  /tmp/score_3d.json"
echo "DONE."
