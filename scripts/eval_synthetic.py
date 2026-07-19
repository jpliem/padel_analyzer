#!/usr/bin/env python
"""Legacy point-target harness for synthetic PadelVic CSV files.

Important: PadelVic describes these clips as Xsens motion-capture renders and
does not identify the CSV coordinates as ball centers. Inspection shows that
they align with the player's ground/root position. The command therefore
refuses known PadelVic CSV files by default so they cannot produce bogus ball
accuracy numbers. Use `eval_ball_labels.py` with reviewed ball labels instead.

CSV format (semicolon-separated, no header):  orig_frame ; ball_px_x ; ball_px_y
The clip is subsampled so CSV row i corresponds 1:1 to clip frame i.

Examples:
    python scripts/eval_synthetic.py \
        --clip data/datasets/padelvic/synthetic/001-1250-17462.mkv \
        --csv  data/datasets/padelvic/synthetic/001-1250-17462.csv \
        --detector yolo

    # quick check, first 500 frames
    python scripts/eval_synthetic.py --clip <mkv> --csv <csv> --max-frames 500
"""
import sys
import os
import argparse
import json
import math
import time

# backend/src on path so we can reuse the real detectors.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))

import cv2  # noqa: E402

# PCK-style thresholds in pixels: fraction of detections within each radius.
PCK_THRESHOLDS_PX = (5, 10, 25, 50)


def load_gt(csv_path: str) -> list[tuple[float, float] | None]:
    """Load ground-truth ball pixel positions in clip-frame order.

    Returns a list indexed by clip frame; each entry is (x, y) or None if the
    row is malformed / ball absent.
    """
    gt: list[tuple[float, float] | None] = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", ";").split(";")
            if len(parts) < 3:
                gt.append(None)
                continue
            try:
                gt.append((float(parts[1]), float(parts[2])))
            except ValueError:
                gt.append(None)
    return gt


def build_detector(detector_type: str, tracknet_model_path: str = "models/tracknet_padel.pt",
                   tracknet_conf: float = 0.3, yolo_fallback: bool = True):
    """Construct a ball detector with the same wiring VideoAnalyzer uses."""
    from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector
    from cv.detectors.tracknet import TrackNetBallDetector
    from cv.detectors.fast_ball import FastBallDetector

    if detector_type == "tracknet":
        yolo = (YoloBallDetector(UnifiedYoloDetector()) if yolo_fallback else None)
        return TrackNetBallDetector(
            model_path=tracknet_model_path,
            conf_threshold=tracknet_conf,
            yolo_fallback=yolo,
        )
    if detector_type == "fast" and not yolo_fallback:
        return FastBallDetector(yolo_fallback=None)
    unified = UnifiedYoloDetector()
    yolo = YoloBallDetector(unified)
    if detector_type == "fast":
        return FastBallDetector(yolo_fallback=yolo)
    return yolo


def bbox_center(bbox) -> tuple[float, float] | None:
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = bbox[:4]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure ball-detection accuracy vs PADELVIC synthetic GT.")
    parser.add_argument("--clip", required=True, help="Synthetic video (.mkv/.mp4)")
    parser.add_argument("--csv", required=True, help="Paired ground-truth CSV")
    parser.add_argument("--detector", choices=["yolo", "tracknet", "fast"],
                        default="yolo")
    parser.add_argument("--tracknet-model", default="models/tracknet_padel.pt",
                        help="TrackNet weights, relative to backend/ unless absolute")
    parser.add_argument("--tracknet-conf", type=float, default=0.3)
    parser.add_argument("--no-yolo-fallback", action="store_true",
                        help="Disable YOLO fallback for TrackNet early/low-confidence frames")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--out", help="Write full per-frame + summary JSON here")
    parser.add_argument("--force-unverified-gt", action="store_true",
                        help="Run despite unverified target semantics (diagnostics only)")
    args = parser.parse_args()

    for p in (args.clip, args.csv):
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1

    normalised_csv = args.csv.replace("\\", "/").lower()
    if ("padelvic/synthetic/" in normalised_csv and
            not args.force_unverified_gt):
        print(
            "ERROR: PadelVic synthetic CSV coordinates are Xsens positional "
            "ground truth, not verified ball centers. Use reviewed labels.json "
            "with scripts/eval_ball_labels.py. Pass --force-unverified-gt only "
            "for target-agnostic diagnostics.", file=sys.stderr)
        return 2

    # Detector/model paths in the code are relative to backend/.
    os.chdir(os.path.join(_ROOT, "backend"))
    clip = args.clip if os.path.isabs(args.clip) else os.path.join(_ROOT, args.clip)
    csv = args.csv if os.path.isabs(args.csv) else os.path.join(_ROOT, args.csv)

    gt = load_gt(csv)
    cap = cv2.VideoCapture(clip)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n_frames != len(gt):
        print(f"WARN: clip frames ({n_frames}) != GT rows ({len(gt)}); "
              f"comparing on the overlap by index.", file=sys.stderr)

    detector = build_detector(
        args.detector,
        tracknet_model_path=args.tracknet_model,
        tracknet_conf=args.tracknet_conf,
        yolo_fallback=not args.no_yolo_fallback,
    )

    errors: list[float] = []          # pixel error on frames where GT and detection both exist
    n_gt = 0                          # frames with a GT ball
    n_detected_when_gt = 0            # GT present AND detector fired
    n_detected_total = 0             # detector fired (any frame)
    per_frame = []

    limit = min(args.max_frames or n_frames, len(gt))
    t0 = time.time()
    frame_no = 0
    while frame_no < limit:
        ret, frame = cap.read()
        if not ret:
            break
        bbox = detector.detect(frame, frame_no)
        pred = bbox_center(bbox)
        truth = gt[frame_no] if frame_no < len(gt) else None

        if pred is not None:
            n_detected_total += 1
        if truth is not None:
            n_gt += 1
            if pred is not None:
                n_detected_when_gt += 1
                err = math.dist(pred, truth)
                errors.append(err)
                per_frame.append({"frame": frame_no, "pred": pred,
                                  "gt": truth, "error_px": round(err, 2)})
            else:
                per_frame.append({"frame": frame_no, "pred": None,
                                  "gt": truth, "error_px": None})

        frame_no += 1
        if frame_no % 100 == 0:
            print(f"\r  {frame_no}/{limit} frames", end="", file=sys.stderr, flush=True)
    cap.release()
    print(file=sys.stderr)
    elapsed = time.time() - t0

    errors.sort()

    def pct(p: float) -> float | None:
        if not errors:
            return None
        idx = min(int(p / 100 * len(errors)), len(errors) - 1)
        return round(errors[idx], 2)

    summary = {
        "clip": os.path.basename(clip),
        "detector": args.detector,
        "tracknet_model": args.tracknet_model if args.detector == "tracknet" else None,
        "tracknet_conf": args.tracknet_conf if args.detector == "tracknet" else None,
        "frames_evaluated": frame_no,
        "gt_frames": n_gt,
        "detection_rate": round(n_detected_when_gt / n_gt, 4) if n_gt else None,
        "mean_error_px": round(sum(errors) / len(errors), 2) if errors else None,
        "median_error_px": pct(50),
        "p90_error_px": pct(90),
        "rmse_px": round(math.sqrt(sum(e * e for e in errors) / len(errors)), 2)
                   if errors else None,
        "pck": {f"<{t}px": round(sum(1 for e in errors if e <= t) / len(errors), 4)
                for t in PCK_THRESHOLDS_PX} if errors else {},
        "fps": round(frame_no / elapsed, 1) if elapsed else 0,
    }

    print("\n=== ball-detection accuracy ===")
    for k, v in summary.items():
        if k == "pck":
            print(f"  PCK (within radius):")
            for tk, tv in v.items():
                print(f"      {tk}: {tv:.1%}")
        elif isinstance(v, float) and k == "detection_rate":
            print(f"  {k}: {v:.1%}")
        else:
            print(f"  {k}: {v}")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        with open(out, "w") as f:
            json.dump({"summary": summary, "per_frame": per_frame}, f)
        print(f"\n  full report -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
