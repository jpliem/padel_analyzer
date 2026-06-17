#!/usr/bin/env python
"""Colour+motion ball candidates + cross-camera epipolar matching -> 3D.

#1 (colour+motion) gives a few ball-coloured MOVING candidates per camera
(excludes static court lines and non-ball movers). #2 (epipolar/geometry):
among all candidate pairs (one per camera), the REAL ball is the pair whose
two sightlines actually intersect (low reprojection error) AND lands on-court
at a plausible height. Heads get excluded by colour; remaining noise gets
excluded by cross-camera geometry. No single camera has to be right alone.

Example:
    python scripts/triangulate_matched.py --start 2 --window 40 --rate 25 --out /tmp/tri_matched.json
"""
import sys, os, argparse, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))
import cv2, numpy as np
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
from ball_motion_color import ball_candidates


def build_cam(video, calib):
    from cv.camera_model import CameraModel
    d = json.load(open(calib)); cam = CameraModel()
    cam.calibrate(d["camera_model_keypoints"], image_width=d.get("image_width", 1920),
                  image_height=d.get("image_height", 1080))
    return cam


def read3(cap, fno):
    out = []
    for f in (fno - 1, fno, fno + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0)); ok, im = cap.read()
        out.append(im if ok else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-a", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    ap.add_argument("--calib-a", default="/tmp/panasonic_cammodel2.json")
    ap.add_argument("--video-b", default="data/datasets/padelvic/cameras/gopro.mp4")
    ap.add_argument("--calib-b", default="/tmp/gopro_cammodel.json")
    ap.add_argument("--sync", default="/tmp/sync_pana_gopro.json")
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=40.0)
    ap.add_argument("--rate", type=float, default=25.0)
    ap.add_argument("--max-reproj", type=float, default=15.0)
    ap.add_argument("--hsv-lo", default="22,50,110")
    ap.add_argument("--hsv-hi", default="48,255,255")
    ap.add_argument("--out", default="/tmp/tri_matched.json")
    args = ap.parse_args()
    lo = [int(v) for v in args.hsv_lo.split(",")]; hi = [int(v) for v in args.hsv_hi.split(",")]

    os.chdir(os.path.join(_ROOT, "backend"))
    from cv.triangulation import triangulate, reprojection_errors
    va = os.path.join(_ROOT, args.video_a); vb = os.path.join(_ROOT, args.video_b)
    cam_a, cam_b = build_cam(va, args.calib_a), build_cam(vb, args.calib_b)
    Pa, Pb = cam_a.projection_matrix(), cam_b.projection_matrix()
    offset = json.load(open(args.sync)).get("offset_seconds", 0.0) if args.sync else 0.0
    capa, capb = cv2.VideoCapture(va), cv2.VideoCapture(vb)
    fpsa = capa.get(cv2.CAP_PROP_FPS) or 50.0; fpsb = capb.get(cv2.CAP_PROP_FPS) or 60.0

    n = int(args.window * args.rate)
    pts = []; matched = 0
    for i in range(n):
        t = args.start + i / args.rate
        pa = read3(capa, int(t * fpsa)); pb = read3(capb, int((t - offset) * fpsb))
        if pa[1] is None or pb[1] is None:
            break
        ca = ball_candidates(pa[0], pa[1], pa[2], lo, hi, 2.0, 400.0)
        cb = ball_candidates(pb[0], pb[1], pb[2], lo, hi, 2.0, 400.0)
        if not ca or not cb:
            continue
        best = None
        for (xa, ya, _) in ca:
            for (xb, yb, _) in cb:
                X = triangulate([(Pa, (xa, ya)), (Pb, (xb, yb))])
                if X is None:
                    continue
                err = max(reprojection_errors(X, [(Pa, (xa, ya)), (Pb, (xb, yb))]))
                x, y, z = float(X[0]), float(X[1]), float(X[2])
                onc = (-1 <= x <= 11) and (-1 <= y <= 21) and (-0.3 <= z <= 8)
                if err <= args.max_reproj and onc and (best is None or err < best[3]):
                    best = (x, y, z, err)
        if best:
            matched += 1
            pts.append({"t": round(t, 3), "x": round(best[0], 2), "y": round(best[1], 2),
                        "z": round(best[2], 2), "reproj_px": round(best[3], 1), "on_court": True})
        if i % 100 == 0:
            print(f"\r  t={t:.1f}s matched={matched}", end="", file=sys.stderr, flush=True)
    capa.release(); capb.release(); print(file=sys.stderr)

    zs = [p["z"] for p in pts]
    print("\n=== colour+motion candidates + epipolar match -> 3D ===")
    print(f"  samples {n} @ {args.rate}Hz")
    print(f"  matched ball (both cams agree, on-court): {matched}")
    if zs:
        print(f"  ball height z: min {min(zs):.2f} median {np.median(zs):.2f} max {max(zs):.2f} m")
        spread = (max(p['x'] for p in pts)-min(p['x'] for p in pts),
                  max(p['y'] for p in pts)-min(p['y'] for p in pts))
        print(f"  x,y spread: {spread[0]:.1f} x {spread[1]:.1f} m  (wide = moving ball, tiny = stuck on a head)")
    json.dump({"cam_a": "panasonic", "cam_b": "gopro", "points": pts}, open(args.out, "w"))
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
