# Ball Detection Diagnostics

Status: captured 2026-06-19 from PADELVIC cameras.

## Purpose

Use these diagnostics before returning to multi-camera triangulation or scoring. They show whether each camera can produce reliable 2D ball candidates. Yellow circles are all colour+motion candidates; the red circle is the continuity-selected 2D candidate.

## Current Finding

Ball detection is not reliable enough yet. Panasonic and GoPro select a candidate almost every frame, but many selections land on rackets, players, glass, fence posts, or reflections. Samsung and iPhone produce fewer candidates, but those views are oblique fence views and are not currently calibrated for 3D.

Fresh 20 s outputs:

| Output | Frames | Selected frames | Avg candidates/frame |
| --- | ---: | ---: | ---: |
| `/tmp/single_cam_panasonic_ball_overlay.mp4` | 1000 | 983 | 4.39 |
| `/tmp/single_cam_gopro_ball_overlay.mp4` | 1199 | 1198 | 7.34 |
| `/tmp/single_cam_samsung_ball_overlay.mp4` | 1200 | 586 | 0.95 |
| `/tmp/single_cam_iphone_ball_overlay.mp4` | 1199 | 715 | 1.03 |
| `/tmp/multiview_panasonic_gopro_ball_overlay.mp4` | 500 | panasonic 492, gopro 500 | panasonic 4.51, gopro 7.43 |

## Reproduce

```bash
PY=backend/venv/bin/python

$PY scripts/annotate_single_cam_ball.py \
  --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
  --start 2 --window 20 --width 1280 \
  --out /tmp/single_cam_panasonic_ball_overlay.mp4

$PY scripts/annotate_single_cam_ball.py \
  --video data/datasets/padelvic/cameras/gopro.mp4 \
  --start 2 --window 20 --width 1280 \
  --out /tmp/single_cam_gopro_ball_overlay.mp4

$PY scripts/annotate_single_cam_ball.py \
  --video data/datasets/padelvic/cameras/samsung.mp4 \
  --start 2 --window 20 --width 1280 \
  --out /tmp/single_cam_samsung_ball_overlay.mp4

$PY scripts/annotate_single_cam_ball.py \
  --video data/datasets/padelvic/cameras/iphone.mp4 \
  --start 2 --window 20 --width 1280 \
  --out /tmp/single_cam_iphone_ball_overlay.mp4

$PY scripts/annotate_multiview_ball.py \
  --video-a data/datasets/padelvic/cameras/panasonic_final.mp4 \
  --video-b data/datasets/padelvic/cameras/gopro.mp4 \
  --label-a panasonic --label-b gopro \
  --sync /tmp/sync_pana_gopro.json \
  --start 2 --window 20 --rate 25 --height 540 \
  --out /tmp/multiview_panasonic_gopro_ball_overlay.mp4
```

## Next Step

Create a real-frame labeled set from Panasonic first, including hard negatives where the detector currently fails. Use `scripts/eval_ball_labels.py` as the gate for every detector change. Do not resume 3D/scoring work until single-camera 2D ball precision and recall are visibly and numerically acceptable.
