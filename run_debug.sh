#!/usr/bin/env bash
#
# One-shot triangulation debug runner. Run from anywhere:
#     bash /Users/jonathan/Documents/Github/padel_analyzer/run_debug.sh
#
# Does, in order:
#   1. gated quality stats on the existing triangulation result (instant)
#   2. visual debug images (court map + height plot + reprojected overlays)
#   3. a fresh triangulation run WITH reprojection gating (~5 min)
#   4. gated stats on the fresh run
# Outputs land in /tmp; paths are printed at the end.

set -uo pipefail

ROOT="/Users/jonathan/Documents/Github/padel_analyzer"
PY="$ROOT/venv/bin/python"
[ -x "$PY" ] || PY="$ROOT/backend/venv/bin/python"
cd "$ROOT"

echo "════════════════════════════════════════════════"
echo " 1) GATED STATS on existing /tmp/triangulated.json"
echo "════════════════════════════════════════════════"
"$PY" - <<'EOF' 2>&1 | grep -vE "Warning|warn"
import json, numpy as np, os
p = '/tmp/triangulated.json'
if not os.path.exists(p):
    print("  (no /tmp/triangulated.json yet — step 3 will create one)"); raise SystemExit
d = json.load(open(p))['points']
for thr in [None, 30, 20, 15]:
    pts = [q for q in d if thr is None or q['reproj_px'] < thr]
    if not pts: continue
    onc = sum(1 for q in pts if -1<=q['x']<=11 and -1<=q['y']<=21 and -0.5<=q['z']<=8)
    zs = [q['z'] for q in pts if -0.5<=q['z']<=8 and -1<=q['y']<=21 and -1<=q['x']<=11]
    lab = 'ALL' if thr is None else f'reproj<{thr}px'
    print(f"  {lab:12}: {len(pts):3d} pts, on-court {100*onc/len(pts):.0f}%, z-median {np.median(zs) if zs else 0:.2f}m")
EOF

echo
echo "════════════════════════════════════════════════"
echo " 2) VISUAL DEBUG IMAGES"
echo "════════════════════════════════════════════════"
"$PY" scripts/debug_triangulation.py --samples 4 2>&1 | grep -vE "Warning|warn"

echo
echo "════════════════════════════════════════════════"
echo " 3) FRESH TRIANGULATION WITH GATING (~5 min)"
echo "════════════════════════════════════════════════"
"$PY" scripts/triangulate_ball.py \
  --video-a data/datasets/padelvic/cameras/panasonic_final.mp4 --calib-a /tmp/panasonic_cammodel2.json \
  --video-b data/datasets/padelvic/cameras/gopro.mp4 --auto-court-b \
  --sync /tmp/sync_pana_gopro.json --start 2 --window 40 --rate 10 \
  --max-reproj 20 --out /tmp/tri_gated.json 2>&1 | grep -vE "Warning|warn"

echo
echo "════════════════════════════════════════════════"
echo " 4) GATED STATS on fresh /tmp/tri_gated.json"
echo "════════════════════════════════════════════════"
"$PY" - <<'EOF' 2>&1 | grep -vE "Warning|warn"
import json, numpy as np, os
p = '/tmp/tri_gated.json'
if not os.path.exists(p):
    print("  (no fresh run output)"); raise SystemExit
d = json.load(open(p))['points']
onc = sum(1 for q in d if -1<=q['x']<=11 and -1<=q['y']<=21 and -0.5<=q['z']<=8)
zs = [q['z'] for q in d if -0.5<=q['z']<=8]
print(f"  kept {len(d)} pts, on-court {100*onc/max(len(d),1):.0f}%, z-median {np.median(zs) if zs else 0:.2f}m")
EOF

echo
echo "════════════════════════════════════════════════"
echo " OUTPUTS"
echo "════════════════════════════════════════════════"
echo "  /tmp/tri_overview.png      (top-down court + height-over-time)"
echo "  /tmp/tri_overlay_0..3.png  (3D reprojected onto both camera frames)"
echo "  /tmp/tri_gated.json        (fresh gated triangulation)"
echo "  open them with:  open /tmp/tri_overview.png /tmp/tri_overlay_*.png"
echo "DONE."
