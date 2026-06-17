#!/usr/bin/env python
"""Draw each camera's calibrated court model back onto its own frame.

The acid test for calibration: reproject the known court lines (baselines,
sidelines, service lines, net) through a camera's CameraModel onto a real
frame. If the drawn lines land on the actual painted court lines, calibration
is correct; if they're shifted/skewed, that camera is mis-calibrated and any
triangulation using it is garbage.

Example:
    python scripts/draw_calibration.py
"""
import sys, os, argparse, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "src"))
import cv2, numpy as np

COURT_W, COURT_L, NET_H = 10.0, 20.0, 0.92
SVC_NEAR, SVC_FAR, CTR = 6.95, 13.05, 5.0

# court line segments as 3D endpoints (z=0 ground unless noted)
LINES = [
    ((0,0,0),(COURT_W,0,0)), ((0,COURT_L,0),(COURT_W,COURT_L,0)),       # baselines
    ((0,0,0),(0,COURT_L,0)), ((COURT_W,0,0),(COURT_W,COURT_L,0)),       # sidelines
    ((0,SVC_NEAR,0),(COURT_W,SVC_NEAR,0)), ((0,SVC_FAR,0),(COURT_W,SVC_FAR,0)),  # service lines
    ((CTR,SVC_NEAR,0),(CTR,SVC_FAR,0)),                                  # center service
    ((0,10,0),(COURT_W,10,0)),                                          # net base
]
NET_TOP = [((0,10,NET_H),(COURT_W,10,NET_H)), ((0,10,0),(0,10,NET_H)), ((COURT_W,10,0),(COURT_W,10,NET_H))]


def build(video, calib_path, auto):
    from cv.camera_model import CameraModel
    from cv.court_detector import CourtDetector
    cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
    ok, frame = cap.read(); cap.release()
    w, h = frame.shape[1], frame.shape[0]
    cam = CameraModel()
    if calib_path:
        d = json.load(open(calib_path))
        cam.calibrate(d["camera_model_keypoints"], image_width=d.get("image_width", w),
                      image_height=d.get("image_height", h))
    else:
        kp = CourtDetector().detect(frame)
        cam.calibrate(kp, image_width=w, image_height=h)
    return cam, frame


def draw(cam, frame, label, out):
    def proj(p):
        u, v = cam.court_to_pixel(p[0], p[1], p[2])
        return int(u), int(v)
    for a, b in LINES:
        cv2.line(frame, proj(a), proj(b), (0, 255, 0), 3)
    for a, b in NET_TOP:
        cv2.line(frame, proj(a), proj(b), (255, 0, 0), 3)
    # corner keypoints as dots
    for c in [(0,0,0),(COURT_W,0,0),(COURT_W,COURT_L,0),(0,COURT_L,0)]:
        cv2.circle(frame, proj(c), 12, (0, 0, 255), -1)
    cv2.putText(frame, label, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 255), 3)
    small = cv2.resize(frame, (1280, int(1280 * frame.shape[0] / frame.shape[1])))
    cv2.imwrite(out, small)
    print(f"  -> {out}")


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(os.path.join(ROOT, "backend"))
    cams = [
        ("panasonic", f"{ROOT}/data/datasets/padelvic/cameras/panasonic_final.mp4", "/tmp/panasonic_cammodel2.json", False),
        ("gopro", f"{ROOT}/data/datasets/padelvic/cameras/gopro.mp4", None, True),
    ]
    for name, video, calib, auto in cams:
        try:
            cam, frame = build(video, calib, auto)
            draw(cam, frame, f"{name} calibrated court (green=lines, blue=net, red=corners)",
                 f"/tmp/calib_{name}.png")
        except Exception as e:
            print(f"  {name}: FAILED — {e}")


if __name__ == "__main__":
    main()
