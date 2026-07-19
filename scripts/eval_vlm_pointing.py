#!/usr/bin/env python
"""Evaluate VLM ball *pointing* on labeled crops (auditor redesign probe).

Instead of asking a VLM to judge "is the marker on the ball" (small models
answer with a fixed bias), ask it to point at the ball; the auditor then
compares the pointed location with the tracker's marker in plain code. This
scores pointing accuracy against ground truth on the same crops produced by
eval_vlm_auditor.py --crop.

Usage:
    backend/.venv/bin/python scripts/eval_vlm_pointing.py \
        --report /tmp/vlm_auditor_eval_crop.json --model qwen3-vl:2b-instruct
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import median

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pydantic import Field  # noqa: E402

from vlm_coach.ollama_client import OllamaClient, OllamaError  # noqa: E402
from vlm_coach.schemas import StrictModel  # noqa: E402

POINT_PROMPT = """This is a crop from an indoor padel match video (blue court,
glass walls). Somewhere in this image there may be a small yellow padel ball,
possibly motion-blurred into a streak. Ignore the red circle drawn on the
image — it is an overlay, not the ball.

If you can see the ball, return found=true and its pixel coordinates
(x from left, y from top). If no ball is visible, return found=false."""


class BallPoint(StrictModel):
    found: bool = False
    x: int = Field(default=0, ge=0, le=4096)
    y: int = Field(default=0, ge=0, le=4096)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


def crop_origin(marker_px, image_shape, crop):
    height, width = image_shape[:2]
    half = crop // 2
    left = min(max(marker_px[0] - half, 0), max(width - crop, 0))
    top = min(max(marker_px[1] - half, 0), max(height - crop, 0))
    return left, top


def main() -> int:
    parser = argparse.ArgumentParser(description="VLM ball pointing eval on audit crops")
    parser.add_argument("--report", required=True,
                        help="Report from eval_vlm_auditor.py --crop N")
    parser.add_argument("--model", default="qwen3-vl:2b-instruct")
    parser.add_argument("--threshold-px", type=float, default=24,
                        help="Marker agreement radius in crop pixels")
    parser.add_argument("--out", default="/tmp/vlm_pointing_eval.json")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text())
    crop = int(report.get("crop") or 0)
    if crop <= 0:
        print("report was not produced with --crop", file=sys.stderr)
        return 1

    source_labels = json.loads(Path(report["labels"]).read_text())
    label_by_frame = {l["frame"]: l for l in source_labels["labels"]}

    seen = {}
    for entry in report["models"].values():
        for case in entry["cases"]:
            seen[(case["frame"], case["condition"])] = case
    cases = sorted(seen.values(), key=lambda c: (c["frame"], c["condition"]))

    client = OllamaClient()
    rows = []
    for case in cases:
        label = label_by_frame[case["frame"]]
        scale = float(label.get("image_scale", 1.0))
        # image size varies per source; recover from the crop file itself
        image = cv2.imread(case["image"])
        if image is None:
            continue
        gt_full = [c * scale for c in label["center"]]
        origin = crop_origin(case["marker_px"],
                             (image.shape[0] + 10_000, image.shape[1] + 10_000), crop)
        # origin computed against full frame; marker was clamped there, so use
        # stored marker/gt in full-frame coords and rebase:
        origin = (case["marker_px"][0] - crop / 2, case["marker_px"][1] - crop / 2)
        origin = (max(origin[0], 0), max(origin[1], 0))
        gt_in_crop = (gt_full[0] - origin[0], gt_full[1] - origin[1])
        marker_in_crop = (case["marker_px"][0] - origin[0],
                          case["marker_px"][1] - origin[1])
        try:
            point = client.structured(args.model, POINT_PROMPT, BallPoint,
                                      images=[Path(case["image"])])
        except OllamaError as exc:
            rows.append({**case, "point": None, "error": str(exc)[:200]})
            continue
        error_px = (math.dist((point.x, point.y), gt_in_crop)
                    if point.found else None)
        agrees_with_marker = (point.found and
                              math.dist((point.x, point.y), marker_in_crop)
                              <= args.threshold_px)
        rows.append({
            "frame": case["frame"], "condition": case["condition"],
            "found": point.found,
            "point": [point.x, point.y] if point.found else None,
            "gt_in_crop": [round(v, 1) for v in gt_in_crop],
            "marker_in_crop": [round(v, 1) for v in marker_in_crop],
            "error_px": round(error_px, 1) if error_px is not None else None,
            "verdict_marker_on_ball": bool(agrees_with_marker),
            "expected": case["condition"] == "correct",
            "image": case["image"],
        })
        print(f"frame {case['frame']:>6} {case['condition']:>7}: "
              f"found={point.found} err={error_px and round(error_px)}px "
              f"verdict={'agree' if agrees_with_marker else 'disagree'} "
              f"expected={'agree' if case['condition'] == 'correct' else 'disagree'}")

    judged = [r for r in rows if "error" not in r]
    errors = [r["error_px"] for r in judged if r["error_px"] is not None]
    right = [r for r in judged if r["verdict_marker_on_ball"] == r["expected"]]
    summary = {
        "model": args.model,
        "cases": len(judged),
        "found_rate": round(sum(1 for r in judged if r["found"]) / len(judged), 3)
                      if judged else None,
        "median_point_error_px": round(median(errors), 1) if errors else None,
        "verdict_accuracy": round(len(right) / len(judged), 3) if judged else None,
        "threshold_px": args.threshold_px,
    }
    Path(args.out).write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
