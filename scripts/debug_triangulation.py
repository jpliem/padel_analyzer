#!/usr/bin/env python
"""Visual debug for two-camera ball triangulation.

Produces:
  1) a top-down court map of the triangulated 3D trajectory (colour = height)
     plus height-over-time — shows whether the 3D path is physically sane.
  2) for a few sample moments, both camera frames with the triangulated 3D
     point reprojected back on (red). If the red dot sits on the real ball in
     BOTH views, calibration + sync are good; the gap is the error you see.

Run triangulate_ball.py first (writes the points JSON).
"""
import sys
import os
import argparse
import json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))

import cv2
import numpy as np


def court_overview(points, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on = [p for p in points if p["on_court"]]
    ts = [p["t"] for p in points]
    zs = [p["z"] for p in points]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # top-down court
    ax1.add_patch(plt.Rectangle((0, 0), 10, 20, fill=False, lw=2))
    ax1.axhline(10, color="gray", lw=1)  # net
    if on:
        sc = ax1.scatter([p["x"] for p in on], [p["y"] for p in on],
                         c=[p["z"] for p in on], cmap="viridis", s=25, vmin=0, vmax=4)
        plt.colorbar(sc, ax=ax1, label="height z (m)")
    ax1.set_xlim(-2, 12); ax1.set_ylim(-2, 22)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("y (m)")
    ax1.set_title(f"top-down court — {len(on)}/{len(points)} on-court")
    # height over time
    ax2.plot(ts, zs, ".-", ms=4)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_xlabel("match time (s)"); ax2.set_ylabel("ball height z (m)")
    ax2.set_title("triangulated ball height over time")
    fig.tight_layout(); fig.savefig(out_path, dpi=90)
    print(f"  overview -> {out_path}")


def reproject_overlays(points, args, n_samples=3):
    from cv.camera_model import CameraModel
    from cv.court_detector import CourtDetector

    os.chdir(os.path.join(_ROOT, "backend"))
    va = os.path.join(_ROOT, args.video_a)
    vb = os.path.join(_ROOT, args.video_b)

    cam_a = CameraModel()
    da = json.load(open(args.calib_a))
    cam_a.calibrate(da["camera_model_keypoints"], image_width=da.get("image_width", 1920),
                    image_height=da.get("image_height", 1080))
    capb0 = cv2.VideoCapture(vb); capb0.set(cv2.CAP_PROP_POS_FRAMES, 600)
    ok, fr = capb0.read(); capb0.release()
    cam_b = CameraModel()
    cam_b.calibrate(CourtDetector().detect(fr), image_width=int(fr.shape[1]),
                    image_height=int(fr.shape[0]))

    offset = json.load(open(args.sync)).get("offset_seconds", 0.0) if args.sync else 0.0
    capa, capb = cv2.VideoCapture(va), cv2.VideoCapture(vb)
    fps_a = capa.get(cv2.CAP_PROP_FPS); fps_b = capb.get(cv2.CAP_PROP_FPS)

    # sample across the trajectory
    on = [p for p in points if p["on_court"]]
    pick = on[:: max(1, len(on) // n_samples)][:n_samples] if on else points[:n_samples]
    outs = []
    for k, p in enumerate(pick):
        t = p["t"]
        capa.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps_a))
        capb.set(cv2.CAP_PROP_POS_FRAMES, int((t - offset) * fps_b))
        oka, fa = capa.read(); okb, fb = capb.read()
        if not (oka and okb):
            continue
        ua, va_ = cam_a.court_to_pixel(p["x"], p["y"], p["z"])
        ub, vb_ = cam_b.court_to_pixel(p["x"], p["y"], p["z"])
        for img, (u, v) in [(fa, (ua, va_)), (fb, (ub, vb_))]:
            cv2.circle(img, (int(u), int(v)), 18, (0, 0, 255), 4)
        # scale to common height, stack side by side
        H = 540
        fa = cv2.resize(fa, (int(fa.shape[1] * H / fa.shape[0]), H))
        fb = cv2.resize(fb, (int(fb.shape[1] * H / fb.shape[0]), H))
        combo = np.hstack([fa, fb])
        cv2.putText(combo, f"t={t}s  3D=({p['x']},{p['y']},{p['z']})m  reproj={p['reproj_px']}px",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        out = f"/tmp/tri_overlay_{k}.png"
        cv2.imwrite(out, combo); outs.append(out)
        print(f"  overlay t={t}s -> {out}")
    capa.release(); capb.release()
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="/tmp/triangulated.json")
    ap.add_argument("--video-a", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    ap.add_argument("--video-b", default="data/datasets/padelvic/cameras/gopro.mp4")
    ap.add_argument("--calib-a", default="/tmp/panasonic_cammodel2.json")
    ap.add_argument("--sync", default="/tmp/sync_pana_gopro.json")
    ap.add_argument("--samples", type=int, default=3)
    args = ap.parse_args()

    data = json.load(open(args.points))
    pts = data["points"]
    print(f"loaded {len(pts)} triangulated points")
    court_overview(pts, "/tmp/tri_overview.png")
    reproject_overlays(pts, args, args.samples)


if __name__ == "__main__":
    main()
