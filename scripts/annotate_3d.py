#!/usr/bin/env python
"""Draw the triangulated 3D ball (with real height) back onto a camera video.

Reads a triangulate_ball.py output JSON, reprojects each on-court 3D point onto
the chosen camera via its CameraModel, and renders a video with the ball
marker, its height label (z in metres), and a fading 3D trail. Visual proof
that the two-camera 3D is real and sits on the ball.

Example:
    python scripts/annotate_3d.py --points /tmp/tri_gated.json \
        --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
        --calib /tmp/panasonic_cammodel2.json --start 2 --window 20 \
        --out /tmp/ball_3d.mp4
"""
import sys, os, argparse, json

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))
import cv2, numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="/tmp/tri_gated.json")
    ap.add_argument("--video", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    ap.add_argument("--calib", default="/tmp/panasonic_cammodel2.json")
    ap.add_argument("--start", type=float, default=2.0)
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--out", default="/tmp/ball_3d.mp4")
    ap.add_argument("--width", type=int, default=1280, help="output width (downscaled)")
    args = ap.parse_args()

    os.chdir(os.path.join(_ROOT, "backend"))
    from cv.camera_model import CameraModel

    video = os.path.join(_ROOT, args.video) if not os.path.isabs(args.video) else args.video
    cal = json.load(open(args.calib))
    cam = CameraModel()
    cam.calibrate(cal["camera_model_keypoints"],
                  image_width=cal.get("image_width", 1920),
                  image_height=cal.get("image_height", 1080))

    pts = json.load(open(args.points))["points"]
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / W
    out_w, out_h = args.width, int(H * scale)

    # index on-court points by source frame number
    by_frame = {}
    for p in pts:
        if p["on_court"]:
            by_frame[int(round(p["t"] * fps))] = p

    writer = None
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*codec), 30, (out_w, out_h))
        if writer.isOpened():
            break
        writer.release(); writer = None

    f0 = int(args.start * fps); f1 = int((args.start + args.window) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    trail = []  # recent (px,py,z)
    drawn = 0
    for fno in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        # nearest point within +-3 frames
        hit = None
        for d in (0, -1, 1, -2, 2, -3, 3):
            if fno + d in by_frame:
                hit = by_frame[fno + d]; break
        if hit is not None:
            u, v = cam.court_to_pixel(hit["x"], hit["y"], hit["z"])
            trail.append((u, v, hit["z"]))
            if len(trail) > 12:
                trail.pop(0)
            drawn += 1
        # draw fading trail
        for i, (u, v, z) in enumerate(trail):
            a = (i + 1) / len(trail)
            cv2.circle(frame, (int(u), int(v)), int(6 + 6 * a), (0, int(180 * a), int(255 * a)), -1)
        if trail:
            u, v, z = trail[-1]
            cv2.circle(frame, (int(u), int(v)), 16, (0, 0, 255), 3)
            cv2.putText(frame, f"ball z={z:.2f}m", (int(u) + 20, int(v) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(frame, f"t={fno/fps:.1f}s  (two-camera triangulated 3D)", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
        writer.write(cv2.resize(frame, (out_w, out_h)))
    cap.release(); writer.release()
    print(f"wrote {args.out}  ({drawn} ball frames drawn)")
    print(f"  open {args.out}")


if __name__ == "__main__":
    main()
