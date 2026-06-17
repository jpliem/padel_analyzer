#!/usr/bin/env python
"""Probe optic-yellow ball candidates by colour, to tune the HSV range.

Padel balls are bright optic-yellow; heads are skin/grey. Colour-segment the
frame for ball-coloured small round blobs and draw every candidate, so we can
confirm the ball is found and heads are excluded, then lock the HSV range.

Example (tune --hsv-lo/--hsv-hi until only the ball + ball-coloured blobs show):
    python scripts/ball_color_probe.py --t 12 --hsv-lo 25,60,120 --hsv-hi 45,255,255
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "src"))
import cv2, numpy as np

ROOT = "/Users/jonathan/Documents/Github/padel_analyzer"


def candidates(frame, lo, hi, amin, amax):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if not (amin <= a <= amax):
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0:
            continue
        circularity = a / (np.pi * r * r)   # 1.0 = perfect circle
        if circularity < 0.5:
            continue
        out.append((x, y, r, a))
    return out, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=12.0, help="match time (s)")
    ap.add_argument("--hsv-lo", default="25,60,120")
    ap.add_argument("--hsv-hi", default="45,255,255")
    ap.add_argument("--amin", type=float, default=4.0)
    ap.add_argument("--amax", type=float, default=600.0)
    args = ap.parse_args()
    lo = [int(v) for v in args.hsv_lo.split(",")]
    hi = [int(v) for v in args.hsv_hi.split(",")]

    cams = [("panasonic", f"{ROOT}/data/datasets/padelvic/cameras/panasonic_final.mp4", 50.0),
            ("gopro", f"{ROOT}/data/datasets/padelvic/cameras/gopro.mp4", 59.94)]
    for name, video, fps in cams:
        cap = cv2.VideoCapture(video); cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.t * fps))
        ok, f = cap.read(); cap.release()
        if not ok:
            print(f"{name}: no frame"); continue
        cand, _ = candidates(f, lo, hi, args.amin, args.amax)
        for x, y, r, a in cand:
            cv2.circle(f, (int(x), int(y)), max(int(r) + 6, 12), (0, 0, 255), 3)
        cv2.putText(f, f"{name} t={args.t}s  {len(cand)} ball-colour candidates",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        out = f"/tmp/colorprobe_{name}.png"
        cv2.imwrite(out, cv2.resize(f, (1280, int(1280 * f.shape[0] / f.shape[1]))))
        print(f"{name}: {len(cand)} candidates -> {out}")


if __name__ == "__main__":
    main()
