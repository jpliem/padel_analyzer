# Public demo datasets

This project uses cross-sport datasets to test individual engineering components.
They do not replace padel footage for final accuracy claims or padel rules.

## Downloaded samples

### OpenTTGames `test_2`

- Local video: `data/datasets/openttgames/test_2/test_2.mp4`
- Local annotations: `data/datasets/openttgames/test_2/annotations/`
- Source: <https://lab.osai.ai/>
- License: CC BY-NC-SA 4.0
- Media: 30 seconds, 1920x1080, 120 FPS
- Labels: 779 ball-position frames and 50 events (28 bounce, 21 net, 1
  empty event)
- Purpose: develop temporal ball localization and the generic hit/bounce/net
  event interface.

The table-tennis event labels are not padel ground truth. In particular, this
sample cannot teach glass rebounds, padel service rules, court exits, or point
scoring. The non-commercial license also means that it must not be included in
commercial marketing or redistributed as commercial training data.

### CalTennis mini synchronized pair

- Local session: `data/datasets/caltennis/01_23_2026_17_00_court2/`
- Local calibration: `data/datasets/caltennis/camera_calibration/01_23_2026_17_00_court2/`
- Local index: `data/datasets/caltennis/metadata_mini.jsonl`
- Source: <https://huggingface.co/datasets/demalenk/caltennis>
- License: CC BY-NC 4.0
- Cameras: opposing east and west baseline views at approximately 60 Hz
- East video: 290,640 encoded frames, 80:46 duration
- West video: 252,206 encoded frames, 70:08 duration
- Alignment: the east recording starts 30 seconds before the west recording,
  as encoded in the video IDs. Matching visual checks confirm that east
  `00:03:00` corresponds to west `00:02:30`. Apply this start offset before
  matching the supplied per-frame relative timestamps.
- Purpose: validate timestamp alignment, camera projection, cross-view
  association, triangulation, and reprojection-error diagnostics.

CalTennis is primarily a human-pose dataset and does not provide the padel ball
labels required by this project. Any ball labels needed for the engineering
test must be added locally. A successful tennis reconstruction demonstrates the
multi-camera machinery, not padel accuracy.

#### Implemented CalTennis proof

The repository now includes:

- `backend/src/cv/caltennis.py`: loads the mini index, relative per-frame
  timestamps, camera intrinsics/extrinsics, shared session time, and overlap.
- `scripts/scan_caltennis_activity.py`: quickly searches a long stream for
  candidate fast yellow-object motion.
- `scripts/caltennis_multiview_demo.py`: combines motion/colour and optional
  tennis TrackNet proposals with calibration, residual frame synchronization,
  triangulation, unique-frame enforcement, reprojection diagnostics, labels,
  and visual previews.

The visually reviewed proof is reproduced with:

```bash
backend/.venv/bin/python scripts/caltennis_multiview_demo.py \
  --start 149.7 --duration 0.5 --rate 60 \
  --hsv-low 25,90,140 --hsv-high 42,255,255 \
  --detector color --max-reprojection 3 --sync-search-frames 2 \
  --review-status visually_verified \
  --out data/experiments/caltennis_multiview_demo_final
```

Current verified result:

- 2 unique two-camera 3D samples / 4 valid 2D labels.
- Median worst-camera reprojection error: 0.51 pixels.
- Reconstructed ball speed between samples: 12.4 m/s.
- Camera B needed a residual alignment search in addition to the 30-second
  recording-start offset.
- A large stationary ball in the foreground remained unselected.

Outputs are written to the ignored local directory
`data/experiments/caltennis_multiview_demo_final/`:

- `labels.json`: original-pixel, per-camera labels.
- `triangulation.json`: pixels, frames, 3D points, sync deltas, and errors.
- `summary.json`: aggregate metrics.
- `frames/`: original per-camera frames referenced by the label document.
- `previews/`: side-by-side visual review evidence.

The `visually_verified` option must only be used after every emitted preview is
inspected. Detector-only runs should retain the default
`provisional_cross_view_geometry` status. Low reprojection error by itself is
not proof of a ball: early trials matched the same lime racket or background
feature in both views, and label validation also caught an invalid reused camera
frame. Both failure modes are now kept out of the final artifact.

### Wikimedia Commons padel video

- Local video: `data/datasets/public_padel/padel_nations_cup_cc_by_3.webm`
- Source: <https://commons.wikimedia.org/wiki/File:Padel_Nations_Cup_moet_harten_veroveren_op_Plein_%2744.webm>
- Creator: RN7
- License: CC BY 3.0
- Purpose: camera-angle and environment robustness testing, and an attributed
  outward-facing recording demo.

This is not synchronized multi-camera footage and must not be used to claim 3D
triangulation performance.

## Correct use in the product roadmap

1. Use OpenTTGames to exercise generic temporal ball and event code.
2. Use CalTennis to exercise synchronization, calibration, and triangulation.
3. Use public and locally authorized padel footage for padel-specific detection,
   active-ball selection, glass/wall behavior, rules, scoring, and final metrics.
4. Keep research-only and non-commercial media out of commercial demonstrations
   unless separate permission is obtained.
