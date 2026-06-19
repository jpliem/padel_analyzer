#!/usr/bin/env python
"""Visualize model-based ball detector outputs on sampled video frames.

This is separate from `triangulate_matched.py`, which uses colour+motion
candidates. Use it to check whether TrackNet or YOLO is actually finding the
padel ball, or confusing rackets/heads/other objects.

Example:
    python scripts/debug_ball_detectors.py --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
        --times 4.5,9.7,12.6 --out-dir /tmp/ball_detector_debug
"""
import argparse
import os
import sys

import cv2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))


def _draw_box(frame, bbox, color, label):
    if not bbox:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(frame, label, (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)


def _tracknet_detection(detector, cap, frame_no):
    frames = []
    start = max(0, frame_no - detector.N_INPUT_FRAMES + 1)
    for fno in range(start, frame_no + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    out = None
    for idx, frame in enumerate(frames):
        out = detector.detect(frame, start + idx)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    ap.add_argument("--times", default="4.5,9.7,12.6,13.6,18.4")
    ap.add_argument("--out-dir", default="/tmp/ball_detector_debug")
    ap.add_argument("--width", type=int, default=1280)
    args = ap.parse_args()

    os.chdir(os.path.join(_ROOT, "backend"))
    from cv.detectors.tracknet import TrackNetBallDetector
    from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector

    video = args.video if os.path.isabs(args.video) else os.path.join(_ROOT, args.video)
    os.makedirs(args.out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0

    unified = UnifiedYoloDetector()
    yolo = YoloBallDetector(unified)
    tracknet = TrackNetBallDetector(model_path="models/tracknet_padel.pt", conf_threshold=0.3)

    for t in [float(v) for v in args.times.split(",") if v.strip()]:
        frame_no = int(round(t * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        if not ok:
            print(f"t={t:.2f}: frame read failed")
            continue
        yolo_box = yolo.detect(frame, frame_no)
        tracknet_box = _tracknet_detection(tracknet, cap, frame_no)

        annotated = frame.copy()
        _draw_box(annotated, yolo_box, (0, 255, 255), "YOLO sports-ball")
        _draw_box(annotated, tracknet_box, (0, 0, 255), "TrackNet")
        cv2.putText(annotated, f"t={t:.2f}s frame={frame_no}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        scale = args.width / annotated.shape[1]
        resized = cv2.resize(annotated, (args.width, int(annotated.shape[0] * scale)))
        out = os.path.join(args.out_dir, f"detectors_t{t:.2f}.jpg")
        cv2.imwrite(out, resized)
        print(f"t={t:.2f}: yolo={yolo_box} tracknet={tracknet_box} -> {out}")

    cap.release()


if __name__ == "__main__":
    main()
