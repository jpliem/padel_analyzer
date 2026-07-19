#!/usr/bin/env python
"""Evaluate 2D ball detectors against hand-labeled real PADELVIC frames.

Label JSON schema (v1):
{
  "video": "data/datasets/padelvic/cameras/panasonic_final.mp4",
  "labels": [
    {"frame": 225, "state": "visible", "center": [123.0, 456.0]},
    {"frame": 226, "state": "occluded", "center": null}
  ]
}

Examples:
    python scripts/eval_ball_labels.py --labels data/labels/padelvic_ball_labels.json \
        --detector color_motion --threshold-px 15 --out /tmp/ball_eval.json
"""
import argparse
import json
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import cv2  # noqa: E402


def label_visible(label):
    if "state" in label:
        return label.get("state") in ("visible", "blurred") and label.get("center") is not None
    ball = label.get("ball") or {}
    return bool(ball.get("visible")) and ball.get("x") is not None and ball.get("y") is not None


def label_reviewed(label):
    return label.get("state", "legacy") not in ("unreviewed", "uncertain")


def match_prediction(label, prediction, threshold_px=15.0):
    visible = label_visible(label)
    if not visible:
        return {
            "frame": label.get("frame"),
            "visible": False,
            "pred": prediction,
            "hit": False,
            "false_positive": prediction is not None,
            "error_px": None,
        }

    if "center" in label:
        truth = tuple(float(v) for v in label["center"])
    else:
        truth = (float(label["ball"]["x"]), float(label["ball"]["y"]))
    if prediction is None:
        return {
            "frame": label.get("frame"),
            "visible": True,
            "gt": truth,
            "pred": None,
            "hit": False,
            "false_positive": False,
            "error_px": None,
        }

    error = round(math.dist(truth, prediction), 2)
    return {
        "frame": label.get("frame"),
        "visible": True,
        "gt": truth,
        "pred": prediction,
        "hit": error <= threshold_px,
        "false_positive": False,
        "error_px": error,
    }


def compute_metrics(labels, predictions, threshold_px=15.0):
    labels = [label for label in labels if label_reviewed(label)]
    rows = [
        match_prediction(
            label,
            predictions.get(prediction_key(label),
                            predictions.get(int(label["frame"]))),
            threshold_px,
        )
        for label in labels
    ]
    visible_rows = [r for r in rows if r["visible"]]
    pred_visible_rows = [r for r in visible_rows if r["pred"] is not None]
    hits = [r for r in visible_rows if r["hit"]]
    misses = [r for r in visible_rows if r["pred"] is None]
    wrong = [r for r in visible_rows if r["pred"] is not None and not r["hit"]]
    false_positive_rows = [r for r in rows if r["false_positive"]]
    errors = sorted(r["error_px"] for r in pred_visible_rows if r["error_px"] is not None)

    predicted_total = len(pred_visible_rows) + len(false_positive_rows)
    correct_total = len(hits)
    visible_total = len(visible_rows)

    def percentile(p):
        if not errors:
            return None
        idx = min(int(p / 100.0 * len(errors)), len(errors) - 1)
        return errors[idx]

    summary = {
        "labels": len(labels),
        "visible_frames": visible_total,
        "invisible_frames": len(labels) - visible_total,
        "predicted_visible_frames": len(pred_visible_rows),
        "hits": len(hits),
        "misses": len(misses),
        "wrong_visible_predictions": len(wrong),
        "false_positive_frames": len(false_positive_rows),
        "precision": round(correct_total / predicted_total, 4) if predicted_total else None,
        "recall": round(correct_total / visible_total, 4) if visible_total else None,
        "mean_error_px": round(sum(errors) / len(errors), 2) if errors else None,
        "median_error_px": percentile(50),
        "p90_error_px": percentile(90),
        "threshold_px": threshold_px,
    }
    return summary, rows


def bbox_center(bbox):
    if not bbox or len(bbox) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def build_detector(detector_type, tracknet_model_path="models/tracknet_padel.pt",
                   tracknet_conf=0.3, yolo_fallback=True):
    if detector_type == "color_motion":
        return None

    from cv.detectors.fast_ball import FastBallDetector
    from cv.detectors.tracknet import TrackNetBallDetector
    from cv.detectors.yolo import UnifiedYoloDetector, YoloBallDetector

    unified = UnifiedYoloDetector()
    yolo = YoloBallDetector(unified)
    if detector_type == "yolo":
        return yolo
    if detector_type == "tracknet":
        return TrackNetBallDetector(
            model_path=tracknet_model_path,
            conf_threshold=tracknet_conf,
            yolo_fallback=yolo if yolo_fallback else None,
        )
    if detector_type == "fast":
        return FastBallDetector(yolo_fallback=yolo)
    raise ValueError(f"unknown detector: {detector_type}")


def color_motion_prediction(cap, frame_no, hsv_lo, hsv_hi):
    from ball_motion_color import ball_candidates

    frames = []
    for fno in (frame_no - 1, frame_no, frame_no + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(fno, 0))
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    if any(f is None for f in frames):
        return None
    candidates = ball_candidates(
        frames[0], frames[1], frames[2],
        hsv_lo, hsv_hi, amin=2.0, amax=400.0,
        min_circularity=0.35,
        min_fill_ratio=0.30,
        max_aspect_ratio=2.2,
        min_radius=2.5,
        max_radius=18.0,
        return_metrics=True,
    )
    if not candidates:
        return None
    best = max(candidates, key=lambda c: c["circularity"] * c["fill_ratio"])
    return (float(best["x"]), float(best["y"]))


def tracknet_prediction(cap, detector, frame_no):
    """Evaluate TrackNet with actual consecutive frames ending at frame_no."""
    window_size = int(detector.N_INPUT_FRAMES)
    start_frame = frame_no - window_size + 1
    if start_frame < 0:
        return None
    detector.reset()
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prediction = None
    for current_frame in range(start_frame, frame_no + 1):
        ok, frame = cap.read()
        if not ok:
            return None
        prediction = detector.detect(frame, current_frame)
    return bbox_center(prediction)


def label_video(labels_doc, label):
    """Resolve the source video for a label (multi-source v1 or legacy doc)."""
    video = label.get("source_video") or labels_doc.get("video")
    if not video:
        raise KeyError("label has no source_video and document has no video")
    return video if os.path.isabs(video) else os.path.join(_ROOT, video)


def prediction_key(label):
    return (label.get("source_video") or "", int(label["frame"]))


def run_detector(labels_doc, detector_type, hsv_lo, hsv_hi,
                 tracknet_model_path="models/tracknet_padel.pt",
                 tracknet_conf=0.3, yolo_fallback=True):
    detector = build_detector(detector_type, tracknet_model_path,
                              tracknet_conf, yolo_fallback)
    predictions = {}
    captures = {}

    try:
        for idx, label in enumerate(labels_doc["labels"]):
            frame_no = int(label["frame"])
            video_path = label_video(labels_doc, label)
            cap = captures.get(video_path)
            if cap is None:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise SystemExit(f"cannot open video: {video_path}")
                captures[video_path] = cap
            if detector_type == "color_motion":
                pred = color_motion_prediction(cap, frame_no, hsv_lo, hsv_hi)
            elif detector_type == "tracknet":
                pred = tracknet_prediction(cap, detector, frame_no)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ok, frame = cap.read()
                pred = bbox_center(detector.detect(frame, frame_no)) if ok else None
            predictions[prediction_key(label)] = pred
            if (idx + 1) % 25 == 0:
                print(f"\r  evaluated {idx + 1}/{len(labels_doc['labels'])}", end="", file=sys.stderr)
        print(file=sys.stderr)
    finally:
        for cap in captures.values():
            cap.release()
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--detector", choices=["color_motion", "yolo", "tracknet", "fast"],
                    default="color_motion")
    ap.add_argument("--tracknet-model", default="models/tracknet_padel.pt")
    ap.add_argument("--tracknet-conf", type=float, default=0.3)
    ap.add_argument("--no-yolo-fallback", action="store_true")
    ap.add_argument("--threshold-px", type=float, default=15.0)
    ap.add_argument("--hsv-lo", default="22,50,110")
    ap.add_argument("--hsv-hi", default="48,255,255")
    ap.add_argument(
        "--sequence-id", action="append",
        help="evaluate only the selected sequence/rally (repeatable)",
    )
    ap.add_argument("--split", choices=["train", "val", "test"],
                    help="evaluate only labels in the selected split")
    ap.add_argument("--out")
    args = ap.parse_args()

    labels_path = args.labels if os.path.isabs(args.labels) else os.path.join(_ROOT, args.labels)
    labels_doc = json.load(open(labels_path))
    if args.split:
        labels_doc["labels"] = [
            label for label in labels_doc["labels"]
            if label.get("split") == args.split
        ]
        if not labels_doc["labels"]:
            raise SystemExit("no labels matched --split")
    if args.sequence_id:
        selected = set(args.sequence_id)
        labels_doc["labels"] = [
            label for label in labels_doc["labels"]
            if label.get("sequence_id") in selected
        ]
        if not labels_doc["labels"]:
            raise SystemExit("no labels matched --sequence-id")
    hsv_lo = [int(v) for v in args.hsv_lo.split(",")]
    hsv_hi = [int(v) for v in args.hsv_hi.split(",")]

    os.chdir(os.path.join(_ROOT, "backend"))
    t0 = time.time()
    predictions = run_detector(
        labels_doc,
        args.detector,
        hsv_lo,
        hsv_hi,
        tracknet_model_path=args.tracknet_model,
        tracknet_conf=args.tracknet_conf,
        yolo_fallback=not args.no_yolo_fallback,
    )
    summary, rows = compute_metrics(labels_doc["labels"], predictions, args.threshold_px)
    summary["detector"] = args.detector
    summary["tracknet_model"] = args.tracknet_model if args.detector == "tracknet" else None
    summary["tracknet_conf"] = args.tracknet_conf if args.detector == "tracknet" else None
    summary["fps"] = round(len(labels_doc["labels"]) / max(time.time() - t0, 1e-9), 2)
    summary["split"] = args.split
    summary["sequence_ids"] = sorted({
        label.get("sequence_id") for label in labels_doc["labels"]
        if label.get("sequence_id")
    })

    print("\n=== real ball-label detector accuracy ===")
    for key, value in summary.items():
        if key in {"precision", "recall"} and value is not None:
            print(f"  {key}: {value:.1%}")
        else:
            print(f"  {key}: {value}")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2)
        print(f"\n  report -> {out}")


if __name__ == "__main__":
    main()
