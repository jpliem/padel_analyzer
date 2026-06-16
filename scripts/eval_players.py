#!/usr/bin/env python
"""Accuracy harness — measure player-position error against PADELVIC real-footage GT.

PADELVIC ships `derived/PadelVic_Panasonic_labeling.xlsx`. Its `Positions` sheet
gives, per frame, the 4 players (TL/TR/BL/BR by court quadrant) in BOTH pixels
(`*_x/*_y`) and court meters (`*_xm/*_ym`) for the panasonic camera.

Because each frame supplies 4 ground-plane point correspondences (pixel↔meter),
those 4 points define the ground homography for that frame. We project our
detector's player foot-pixels through that GT homography → land directly in the
GT court-meter frame → compare. This makes the result independent of our own
calibration; it measures detection + foot-point accuracy in interpretable meters.

Frame numbers map 1:1 to cameras/panasonic_final.mp4 (50 fps, 3626x1960).

Example:
    python scripts/eval_players.py \
        --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
        --xlsx  data/datasets/padelvic/derived/PadelVic_Panasonic_labeling.xlsx \
        --max-frames 400
"""
import sys
import os
import argparse
import json
import math
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

QUADRANTS = ("TL", "TR", "BL", "BR")
DEFAULT_MATCH_THRESH_M = 2.0  # a GT player counts as detected if a prediction lands within this


def load_positions(xlsx_path: str) -> dict:
    """Return {frame_int: {'px': 4x2 float32, 'm': 4x2 float32}} for TL,TR,BL,BR."""
    import pandas as pd
    df = pd.read_excel(xlsx_path, sheet_name="Positions")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    out = {}
    for _, row in df.iterrows():
        if math.isnan(row["Frame"]):
            continue
        px = np.array([[row[f"{q}_x"], row[f"{q}_y"]] for q in QUADRANTS], dtype=np.float32)
        m = np.array([[row[f"{q}_xm"], row[f"{q}_ym"]] for q in QUADRANTS], dtype=np.float32)
        if np.isnan(px).any() or np.isnan(m).any():
            continue
        out[int(row["Frame"])] = {"px": px, "m": m}
    return out


def build_player_detector(conf_threshold: float | None = None):
    from cv.detectors.yolo import UnifiedYoloDetector, YoloPlayerDetector
    if conf_threshold is not None:
        return YoloPlayerDetector(UnifiedYoloDetector(), conf_threshold=conf_threshold)
    return YoloPlayerDetector(UnifiedYoloDetector())


def foot_points(detections: np.ndarray) -> np.ndarray:
    """Bottom-center (foot) pixel of each player bbox."""
    if detections is None or len(detections) == 0:
        return np.empty((0, 2), dtype=np.float32)
    cx = (detections[:, 0] + detections[:, 2]) / 2.0
    cy = detections[:, 3]  # bbox bottom = feet
    return np.stack([cx, cy], axis=1).astype(np.float32)


def greedy_match(gt_m: np.ndarray, pred_m: np.ndarray, thresh: float):
    """Match each GT player to the nearest unused prediction within thresh.

    Returns list of (quadrant_index, error_m_or_None).
    """
    used = set()
    results = []
    for gi in range(len(gt_m)):
        best_j, best_d = None, None
        for pj in range(len(pred_m)):
            if pj in used:
                continue
            d = float(math.dist(gt_m[gi], pred_m[pj]))
            if best_d is None or d < best_d:
                best_d, best_j = d, pj
        if best_j is not None and best_d <= thresh:
            used.add(best_j)
            results.append((gi, best_d))
        else:
            results.append((gi, None))
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure player-position accuracy vs PADELVIC GT.")
    ap.add_argument("--video", required=True, help="panasonic_final.mp4")
    ap.add_argument("--xlsx", required=True, help="PadelVic_Panasonic_labeling.xlsx")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="Process at most N GT-labelled frames (quick runs)")
    ap.add_argument("--match-thresh", type=float, default=DEFAULT_MATCH_THRESH_M,
                    help=f"Max meters to count a GT player as detected (default {DEFAULT_MATCH_THRESH_M})")
    ap.add_argument("--player-conf", type=float, default=None,
                    help="Override YOLO player conf threshold (default 0.6)")
    ap.add_argument("--out", help="Write per-frame + summary JSON here")
    args = ap.parse_args()

    video = args.video if os.path.isabs(args.video) else os.path.join(_ROOT, args.video)
    xlsx = args.xlsx if os.path.isabs(args.xlsx) else os.path.join(_ROOT, args.xlsx)
    for p in (video, xlsx):
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1

    # Detector/model paths are relative to backend/.
    os.chdir(os.path.join(_ROOT, "backend"))

    gt = load_positions(xlsx)
    gt_frames = sorted(gt.keys())
    if args.max_frames:
        gt_frames = gt_frames[:args.max_frames]
    target = set(gt_frames)
    last_frame = gt_frames[-1]
    print(f"GT frames: {len(gt_frames)} (range {gt_frames[0]}-{last_frame})")

    detector = build_player_detector(args.player_conf)

    errors = []                              # all matched meter errors
    per_quad = {q: {"errs": [], "gt": 0, "hit": 0} for q in QUADRANTS}
    per_frame = []
    n_processed = 0

    cap = cv2.VideoCapture(video)
    frame_no = -1
    t0 = time.time()
    while True:
        ret = cap.grab()                     # cheap: advance without decoding
        if not ret:
            break
        frame_no += 1
        if frame_no > last_frame:
            break
        if frame_no not in target:
            continue
        ok, frame = cap.retrieve()           # decode only labelled frames
        if not ok:
            continue

        H = cv2.getPerspectiveTransform(gt[frame_no]["px"], gt[frame_no]["m"])
        feet = foot_points(detector.detect(frame, frame_no))
        if len(feet):
            pred_m = cv2.perspectiveTransform(feet.reshape(-1, 1, 2), H).reshape(-1, 2)
        else:
            pred_m = np.empty((0, 2), dtype=np.float32)

        matches = greedy_match(gt[frame_no]["m"], pred_m, args.match_thresh)
        frame_errs = []
        for gi, err in matches:
            q = QUADRANTS[gi]
            per_quad[q]["gt"] += 1
            if err is not None:
                per_quad[q]["hit"] += 1
                per_quad[q]["errs"].append(err)
                errors.append(err)
                frame_errs.append(round(err, 3))
        per_frame.append({"frame": frame_no, "n_pred": len(pred_m),
                          "errors_m": frame_errs})

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"\r  {n_processed}/{len(gt_frames)} GT frames", end="",
                  file=sys.stderr, flush=True)
    cap.release()
    print(file=sys.stderr)
    elapsed = time.time() - t0

    errors.sort()

    def pct(p):
        if not errors:
            return None
        return round(errors[min(int(p / 100 * len(errors)), len(errors) - 1)], 3)

    total_gt = sum(q["gt"] for q in per_quad.values())
    total_hit = sum(q["hit"] for q in per_quad.values())
    summary = {
        "video": os.path.basename(video),
        "gt_frames_processed": n_processed,
        "match_thresh_m": args.match_thresh,
        "detection_rate": round(total_hit / total_gt, 4) if total_gt else None,
        "mean_error_m": round(sum(errors) / len(errors), 3) if errors else None,
        "median_error_m": pct(50),
        "p90_error_m": pct(90),
        "per_quadrant": {
            q: {
                "detection_rate": round(v["hit"] / v["gt"], 4) if v["gt"] else None,
                "median_error_m": round(float(np.median(v["errs"])), 3) if v["errs"] else None,
            } for q, v in per_quad.items()
        },
        "fps": round(n_processed / elapsed, 1) if elapsed else 0,
    }

    print("\n=== player-position accuracy (meters, in GT court frame) ===")
    print(f"  video: {summary['video']}")
    print(f"  GT frames: {summary['gt_frames_processed']}  (match thresh {args.match_thresh} m)")
    print(f"  detection rate: {summary['detection_rate']:.1%}" if summary['detection_rate'] is not None else "  detection rate: n/a")
    print(f"  error m  -> mean {summary['mean_error_m']}  median {summary['median_error_m']}  p90 {summary['p90_error_m']}")
    print("  per quadrant:")
    for q in QUADRANTS:
        pq = summary["per_quadrant"][q]
        dr = f"{pq['detection_rate']:.0%}" if pq['detection_rate'] is not None else "n/a"
        print(f"      {q}: detect {dr}  median_err {pq['median_error_m']} m")
    print(f"  speed: {summary['fps']} fps")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        with open(out, "w") as f:
            json.dump({"summary": summary, "per_frame": per_frame}, f)
        print(f"\n  full report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
