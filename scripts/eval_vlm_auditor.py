#!/usr/bin/env python
"""Benchmark VLM auditor models on labeled ball frames.

Uses reviewed ball labels as ground truth to score how well a local VLM judges
"is the red marker on the ball" — the exact job of vlm_coach.track_audit. For
each sampled labeled frame it renders two variants:

  correct: marker drawn at the labeled ball center (expected verdict yes/close)
  wrong:   marker displaced by a fixed offset (expected verdict no)

Each model under test judges every variant; the report gives per-model accuracy
on both conditions plus unclear rates. This measures the auditor, not the
tracker — verdicts never touch scoring.

Usage (repo root):
    backend/.venv/bin/python scripts/eval_vlm_auditor.py \
        --labels data/labels/padelvic_panasonic_combined/labels.json \
        --models qwen2.5vl:3b qwen3-vl:2b --sample 15
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vlm_coach.ollama_client import OllamaClient, OllamaError  # noqa: E402
from vlm_coach.schemas import TrackAuditVerdict, model_dump  # noqa: E402
from vlm_coach.track_audit import AUDIT_PROMPT  # noqa: E402

WRONG_OFFSET_PX = 140
MARKER_RADIUS = 16
CORRECT_VERDICTS = ("yes", "close")
WRONG_VERDICTS = ("no",)


def load_visible_labels(labels_path: Path, split: str) -> list[dict]:
    doc = json.loads(labels_path.read_text())
    labels = [
        item for item in doc.get("labels", [])
        if item.get("state") == "visible" and item.get("center")
        and (split == "all" or item.get("split") == split)
    ]
    base = labels_path.parent
    resolved = []
    for item in labels:
        image_path = (base / item["image"]).resolve()
        if image_path.exists():
            resolved.append({**item, "image_path": image_path})
    return resolved


def scaled_center(label: dict) -> tuple[float, float]:
    scale = float(label.get("image_scale", 1.0))
    x, y = label["center"]
    return x * scale, y * scale


def wrong_position(center: tuple[float, float], shape, rng: random.Random) -> tuple[float, float]:
    height, width = shape[:2]
    for _ in range(20):
        angle = rng.uniform(0, 2 * math.pi)
        x = center[0] + WRONG_OFFSET_PX * math.cos(angle)
        y = center[1] + WRONG_OFFSET_PX * math.sin(angle)
        if MARKER_RADIUS < x < width - MARKER_RADIUS and MARKER_RADIUS < y < height - MARKER_RADIUS:
            return x, y
    return min(max(center[0] + WRONG_OFFSET_PX, MARKER_RADIUS), width - MARKER_RADIUS), center[1]


def render_marker(image, position: tuple[float, float], crop: int = 0):
    frame = image.copy()
    cv2.circle(frame, (int(position[0]), int(position[1])), MARKER_RADIUS, (0, 0, 255), 4)
    if crop <= 0:
        return frame
    height, width = frame.shape[:2]
    half = crop // 2
    left = int(min(max(position[0] - half, 0), max(width - crop, 0)))
    top = int(min(max(position[1] - half, 0), max(height - crop, 0)))
    return frame[top:top + crop, left:left + crop]


def unload_model(client: OllamaClient, model: str) -> None:
    """Free unified memory before loading the next model (8 GB Macs)."""
    try:
        import httpx
        httpx.post(f"{client.base_url}/api/generate",
                   json={"model": model, "keep_alive": 0}, timeout=30.0)
    except Exception:
        pass


def judge(client: OllamaClient, model: str, image_path: Path) -> dict:
    started = time.monotonic()
    try:
        verdict = client.structured(model, AUDIT_PROMPT, TrackAuditVerdict,
                                    images=[image_path])
        payload = model_dump(verdict)
        error = None
    except OllamaError as exc:
        payload, error = None, str(exc)[:300]
    return {"verdict": payload, "error": error,
            "seconds": round(time.monotonic() - started, 2)}


def score(cases: list[dict]) -> dict:
    def bucket(condition: str) -> dict:
        rows = [c for c in cases if c["condition"] == condition and c["verdict"]]
        expected = CORRECT_VERDICTS if condition == "correct" else WRONG_VERDICTS
        hits = [c for c in rows if c["verdict"]["marker_on_ball"] in expected]
        unclear = [c for c in rows if c["verdict"]["marker_on_ball"] == "unclear"]
        return {
            "frames": len(rows),
            "accuracy": round(len(hits) / len(rows), 3) if rows else None,
            "unclear_rate": round(len(unclear) / len(rows), 3) if rows else None,
        }

    answered = [c for c in cases if c["verdict"]]
    return {
        "correct_marker": bucket("correct"),
        "wrong_marker": bucket("wrong"),
        "errors": len(cases) - len(answered),
        "mean_seconds": round(
            sum(c["seconds"] for c in cases) / len(cases), 2) if cases else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark VLM audit models on ball labels")
    parser.add_argument("--labels", required=True)
    parser.add_argument("--models", nargs="+", default=["qwen2.5vl:3b", "qwen3-vl:2b"])
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--sample", type=int, default=15)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--crop", type=int, default=0,
                        help="Crop size (px) around the marker; 0 = full frame")
    parser.add_argument("--out", default="/tmp/vlm_auditor_eval.json")
    parser.add_argument("--frames-dir", default="/tmp/vlm_auditor_frames")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    labels = load_visible_labels(Path(args.labels), args.split)
    if not labels:
        print("no usable labels found", file=sys.stderr)
        return 1
    if len(labels) > args.sample:
        labels = rng.sample(labels, args.sample)
    labels.sort(key=lambda item: item["frame"])

    frames_dir = Path(args.frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    variants = []
    for label in labels:
        image = cv2.imread(str(label["image_path"]))
        if image is None:
            continue
        center = scaled_center(label)
        for condition, position in (
            ("correct", center),
            ("wrong", wrong_position(center, image.shape, rng)),
        ):
            out_path = frames_dir / f"{label['frame']:06d}_{condition}.jpg"
            cv2.imwrite(str(out_path), render_marker(image, position, crop=args.crop),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
            variants.append({
                "frame": label["frame"], "condition": condition,
                "marker_px": [round(position[0], 1), round(position[1], 1)],
                "gt_px": [round(center[0], 1), round(center[1], 1)],
                "image": str(out_path),
            })

    client = OllamaClient()
    health = client.health()
    if not health["available"]:
        print(f"Ollama unavailable: {health.get('error')}", file=sys.stderr)
        return 1

    report = {"labels": args.labels, "split": args.split, "crop": args.crop,
              "wrong_offset_px": WRONG_OFFSET_PX, "models": {}}
    for index, model in enumerate(args.models):
        if index:
            unload_model(client, args.models[index - 1])
        cases = []
        for variant in variants:
            result = judge(client, model, Path(variant["image"]))
            cases.append({**variant, **result})
            verdict = result["verdict"]["marker_on_ball"] if result["verdict"] else "ERROR"
            print(f"[{model}] frame {variant['frame']:>6} {variant['condition']:>7} "
                  f"-> {verdict:<15} ({result['seconds']}s)")
        report["models"][model] = {"summary": score(cases), "cases": cases}

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {args.out}")
    for model, entry in report["models"].items():
        summary = entry["summary"]
        print(f"{model}: correct-marker acc {summary['correct_marker']['accuracy']} "
              f"| wrong-marker catch {summary['wrong_marker']['accuracy']} "
              f"| unclear {summary['correct_marker']['unclear_rate']}/"
              f"{summary['wrong_marker']['unclear_rate']} "
              f"| {summary['mean_seconds']}s/frame | errors {summary['errors']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
