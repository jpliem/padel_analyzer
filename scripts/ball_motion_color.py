#!/usr/bin/env python
"""Ball candidates = MOVING + ball-coloured + small + round.

Motion (frame-difference) removes static yellow (court lines, signage); colour
removes non-ball movers; size/roundness removes bodies. The intersection should
isolate the ball and exclude heads. Tune and confirm visually.

Example:
    python scripts/ball_motion_color.py --t 12
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend", "src"))
import cv2, numpy as np

ROOT = "/Users/jonathan/Documents/Github/padel_analyzer"


def ball_candidates(prev, cur, nxt, lo, hi, amin, amax, motion_thr=18):
    # motion: pixels that changed between frames (moving object)
    d1 = cv2.absdiff(cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY), cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY))
    d2 = cv2.absdiff(cv2.cvtColor(nxt, cv2.COLOR_BGR2GRAY), cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY))
    motion = cv2.threshold(cv2.min(d1, d2), motion_thr, 255, cv2.THRESH_BINARY)[1]
    # colour: ball-coloured
    hsv = cv2.cvtColor(cur, cv2.COLOR_BGR2HSV)
    color = cv2.inRange(hsv, np.array(lo), np.array(hi))
    mask = cv2.dilate(cv2.bitwise_and(motion, color), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < amin:
            continue
        (x, y), r = cv2.minEnclosingCircle(c)
        if r <= 0 or a > amax:
            continue
        out.append((float(x), float(y), float(r)))
    return out


def read3(cap, fno):
    fr = []
    for f in (fno - 1, fno, fno + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(f, 0)); ok, im = cap.read()
        fr.append(im if ok else None)
    return fr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=12.0)
    ap.add_argument("--hsv-lo", default="22,50,110")
    ap.add_argument("--hsv-hi", default="48,255,255")
    ap.add_argument("--amin", type=float, default=2.0)
    ap.add_argument("--amax", type=float, default=400.0)
    args = ap.parse_args()
    lo = [int(v) for v in args.hsv_lo.split(",")]; hi = [int(v) for v in args.hsv_hi.split(",")]
    for name, video, fps in [("panasonic", f"{ROOT}/data/datasets/padelvic/cameras/panasonic_final.mp4", 50.0),
                             ("gopro", f"{ROOT}/data/datasets/padelvic/cameras/gopro.mp4", 59.94)]:
        cap = cv2.VideoCapture(video)
        p, c, n = read3(cap, int(args.t * fps)); cap.release()
        if c is None:
            print(f"{name}: no frame"); continue
        cand = ball_candidates(p, c, n, lo, hi, args.amin, args.amax)
        for x, y, r in cand:
            cv2.circle(c, (int(x), int(y)), max(int(r) + 8, 14), (0, 0, 255), 3)
        cv2.putText(c, f"{name} t={args.t}s  {len(cand)} moving+ball-colour candidates",
                    (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 255), 3)
        out = f"/tmp/motioncolor_{name}.png"
        cv2.imwrite(out, cv2.resize(c, (1280, int(1280 * c.shape[0] / c.shape[1]))))
        print(f"{name}: {len(cand)} candidates -> {out}")


if __name__ == "__main__":
    main()
