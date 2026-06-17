#!/usr/bin/env python
"""Real multi-camera ball triangulation on PADELVIC.

Ties together the pieces built this session: two calibrated cameras + a time
offset (from sync_cameras.py) → at each synced moment, detect the ball in both
views, triangulate to true 3D (x, y, height), and report how on-court and
physically-plausible the result is — the thing single-camera fundamentally
cannot do (see memory: ball-position-noise-ceiling).

Each camera's calibration comes from either a JSON of `camera_model_keypoints`
(12 court keypoints in pixels) or auto court detection (--auto-court-<a|b>).

Example:
    python scripts/triangulate_ball.py \
        --video-a data/datasets/padelvic/cameras/panasonic_final.mp4 \
        --calib-a /tmp/panasonic_cammodel2.json \
        --video-b data/datasets/padelvic/cameras/gopro.mp4 --auto-court-b \
        --sync /tmp/sync_pana_gopro.json \
        --start 30 --window 60
"""
import sys
import os
import argparse
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def build_camera(video, calib_path, auto_court):
    """Return a calibrated CameraModel for one view."""
    from cv.camera_model import CameraModel
    from cv.court_detector import CourtDetector

    cam = CameraModel()
    w = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    h = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

    if calib_path:
        data = json.load(open(calib_path))
        cam.calibrate(data["camera_model_keypoints"],
                      net_top_2d=data.get("net_top"),
                      image_width=data.get("image_width", w),
                      image_height=data.get("image_height", h))
        return cam, f"keypoints from {os.path.basename(calib_path)}"
    if auto_court:
        cap = cv2.VideoCapture(video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 600)
        ok, frame = cap.read()
        cap.release()
        kp = CourtDetector().detect(frame) if ok else None
        if not kp:
            raise RuntimeError(f"auto court detection failed on {video}")
        cam.calibrate(kp, image_width=w, image_height=h)
        return cam, f"auto-detected {len(kp)} keypoints"
    raise ValueError("provide --calib or --auto-court for this camera")


def make_ball_detector():
    from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector
    from cv.detectors.tracknet import TrackNetBallDetector
    unified = UnifiedYoloDetector()
    det = TrackNetBallDetector(model_path="models/tracknet_padel.pt",
                               conf_threshold=0.3,
                               yolo_fallback=YoloBallDetector(unified))
    from cv.detectors.yolo import YoloPlayerDetector
    return det, YoloPlayerDetector(unified)


def _in_player_head(cx, cy, player_boxes, head_frac=0.35):
    """True if (cx,cy) sits in the upper `head_frac` of any player bbox.

    The bald-spot/head false positive lives at the top of a player box; the
    real ball at contact is lower (racket/body height), so this rejects heads
    without killing most real ball detections.
    """
    for x1, y1, x2, y2 in player_boxes:
        if x1 <= cx <= x2 and y1 <= cy <= y1 + head_frac * (y2 - y1):
            return True
    return False


def ball_pixel(detector, frame, fno, player_boxes=()):
    bbox = detector.detect(frame, fno)
    if not bbox or len(bbox) < 4:
        return None
    cx, cy = (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0
    if _in_player_head(cx, cy, player_boxes):
        return None  # rejected: looks like a player's head, not the ball
    return (cx, cy)


def main() -> int:
    ap = argparse.ArgumentParser(description="Triangulate the ball from two PADELVIC cameras.")
    ap.add_argument("--video-a", required=True)
    ap.add_argument("--video-b", required=True)
    ap.add_argument("--calib-a")
    ap.add_argument("--calib-b")
    ap.add_argument("--auto-court-a", action="store_true")
    ap.add_argument("--auto-court-b", action="store_true")
    ap.add_argument("--sync", help="sync JSON with offset_seconds (b relative to a)")
    ap.add_argument("--start", type=float, default=30.0, help="match time start (s)")
    ap.add_argument("--window", type=float, default=60.0, help="seconds to process")
    ap.add_argument("--rate", type=float, default=10.0, help="samples/sec")
    ap.add_argument("--max-reproj", type=float, default=20.0,
                    help="reject triangulations with reproj error above this (px); "
                    "high error = cameras disagree (mismatched ball / sync slip)")
    ap.add_argument("--out")
    args = ap.parse_args()

    os.chdir(os.path.join(_ROOT, "backend"))
    va = args.video_a if os.path.isabs(args.video_a) else os.path.join(_ROOT, args.video_a)
    vb = args.video_b if os.path.isabs(args.video_b) else os.path.join(_ROOT, args.video_b)

    from cv.triangulation import triangulate, reprojection_errors

    cam_a, da = build_camera(va, args.calib_a, args.auto_court_a)
    cam_b, db = build_camera(vb, args.calib_b, args.auto_court_b)
    Pa, Pb = cam_a.projection_matrix(), cam_b.projection_matrix()
    if Pa is None or Pb is None:
        print("ERROR: a camera failed 3D calibration (no projection matrix)", file=sys.stderr)
        return 1
    print(f"cam A: {da}")
    print(f"cam B: {db}")

    offset = 0.0
    if args.sync:
        offset = json.load(open(args.sync)).get("offset_seconds", 0.0)
    fps_a = cv2.VideoCapture(va).get(cv2.CAP_PROP_FPS) or 50.0
    fps_b = cv2.VideoCapture(vb).get(cv2.CAP_PROP_FPS) or 60.0
    print(f"sync offset {offset:+.3f}s | fps A {fps_a} B {fps_b}")

    capa, capb = cv2.VideoCapture(va), cv2.VideoCapture(vb)
    det_a, players_a = make_ball_detector()
    det_b, players_b = make_ball_detector()

    n = int(args.window * args.rate)
    pts, both_seen, on_court, reproj, rejected = [], 0, 0, [], 0
    for i in range(n):
        t = args.start + i / args.rate
        capa.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps_a))
        capb.set(cv2.CAP_PROP_POS_FRAMES, int((t - offset) * fps_b))
        oka, fa = capa.read()
        okb, fb = capb.read()
        if not (oka and okb):
            break
        boxes_a = players_a.detect(fa, i)
        boxes_b = players_b.detect(fb, i)
        ba = boxes_a[:, :4].tolist() if len(boxes_a) else []
        bb = boxes_b[:, :4].tolist() if len(boxes_b) else []
        pa = ball_pixel(det_a, fa, i, ba)
        pb = ball_pixel(det_b, fb, i, bb)
        if pa is None or pb is None:
            continue
        both_seen += 1
        X = triangulate([(Pa, pa), (Pb, pb)])
        if X is None:
            continue
        x, y, z = float(X[0]), float(X[1]), float(X[2])
        err = max(reprojection_errors(X, [(Pa, pa), (Pb, pb)]))
        # Reject views that disagree: high reprojection error means the two
        # cameras locked onto different things (or a sub-frame sync slip on a
        # fast ball) — the crossed rays give a nonsense 3D point.
        if err > args.max_reproj:
            rejected += 1
            continue
        reproj.append(err)
        onc = (-1 <= x <= 11) and (-1 <= y <= 21) and (-0.5 <= z <= 8)
        if onc:
            on_court += 1
        pts.append({"t": round(t, 3), "x": round(x, 2), "y": round(y, 2),
                    "z": round(z, 2), "reproj_px": round(err, 1), "on_court": onc})
        if i % 50 == 0:
            print(f"\r  t={t:.1f}s both-seen={both_seen} on-court={on_court}",
                  end="", file=sys.stderr, flush=True)
    capa.release(); capb.release()
    print(file=sys.stderr)

    zs = [p["z"] for p in pts if p["on_court"]]
    print("\n=== real two-camera ball triangulation ===")
    print(f"  samples: {n}, ball seen in BOTH: {both_seen}, "
          f"rejected (reproj>{args.max_reproj}px): {rejected}, kept: {len(pts)}")
    if pts:
        print(f"  on-court & plausible: {on_court}/{len(pts)} ({100*on_court/len(pts):.0f}%)")
        print(f"  median reproj error: {np.median(reproj):.1f}px "
              f"(low = views agree = good sync+calib)")
    if zs:
        print(f"  ball height z (on-court): min {min(zs):.2f} median "
              f"{np.median(zs):.2f} max {max(zs):.2f} m  ← real 3D height")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        json.dump({"cam_a": da, "cam_b": db, "offset_s": offset,
                   "points": pts}, open(out, "w"))
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
