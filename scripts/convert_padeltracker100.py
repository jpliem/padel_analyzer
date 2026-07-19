#!/usr/bin/env python
"""Convert PadelTracker100 COCO ball annotations to the v1 ball-label schema.

PadelTracker100 (Zenodo 14653706, CC-BY-4.0) ships COCO-video JSON per match
video: images[] carry file_name "frame_<n>.<ext>", annotations[] carry bbox
[x, y, w, h] in original video pixels. This produces a labels.json compatible
with scripts/train_temporal_ball.py and scripts/eval_ball_labels.py, and
optionally extracts preview frames from the source video.

Usage (after downloading the dataset):
    backend/.venv/bin/python scripts/convert_padeltracker100.py \
        --coco path/to/ball_annotations.json \
        --video path/to/2022_BCN_FinalF_1.mp4 \
        --out data/labels/padeltracker100_final1 \
        --width 1280
Use --no-frames to write the manifest without extracting preview images.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

FRAME_ID_REGEX = re.compile(r"(\d+)")
SPLIT_FRACTIONS = (("train", 0.7), ("val", 0.15), ("test", 0.15))


def frame_id_from_name(file_name: str) -> int:
    match = FRAME_ID_REGEX.search(Path(file_name).stem)
    if not match:
        raise ValueError(f"cannot parse frame id from {file_name!r}")
    return int(match.group(1))


def load_ball_annotations(coco_path: Path, category_name: str) -> list[dict]:
    doc = json.loads(coco_path.read_text())
    categories = {c["id"]: c.get("name", "") for c in doc.get("categories", [])}
    ball_ids = {cid for cid, name in categories.items()
                if category_name.lower() in name.lower()} or set(categories)
    images = {img["id"]: img for img in doc.get("images", [])}
    rows = []
    for ann in doc.get("annotations", []):
        if ann.get("category_id") not in ball_ids or not ann.get("bbox"):
            continue
        image = images.get(ann.get("image_id"))
        if image is None:
            continue
        x, y, w, h = ann["bbox"]
        rows.append({
            "frame": frame_id_from_name(image["file_name"]),
            "center": [float(x) + float(w) / 2.0, float(y) + float(h) / 2.0],
            "width": image.get("width"),
            "height": image.get("height"),
        })
    rows.sort(key=lambda r: r["frame"])
    deduped = {r["frame"]: r for r in rows}
    return list(deduped.values())


def assign_splits(rows: list[dict], chunk: int = 300) -> None:
    """Split by contiguous frame chunks so temporal neighbours never straddle
    train/val/test (same leakage rule as the existing label sets)."""
    if not rows:
        return
    chunks: list[list[dict]] = []
    current = [rows[0]]
    for row in rows[1:]:
        if row["frame"] - current[-1]["frame"] > 60 or len(current) >= chunk:
            chunks.append(current)
            current = [row]
        else:
            current.append(row)
    chunks.append(current)
    total = len(chunks)
    boundaries = []
    start = 0.0
    for name, fraction in SPLIT_FRACTIONS:
        end = start + fraction
        boundaries.append((name, int(round(start * total)), int(round(end * total))))
        start = end
    for index, rows_chunk in enumerate(chunks):
        split = next((name for name, lo, hi in boundaries if lo <= index < hi),
                     SPLIT_FRACTIONS[-1][0])
        for row in rows_chunk:
            row["split"] = split
            row["sequence_id"] = (
                f"{row.get('camera_id', 'padeltracker100')}:"
                f"{rows_chunk[0]['frame']}-{rows_chunk[-1]['frame']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PadelTracker100 COCO -> v1 ball labels")
    parser.add_argument("--coco", required=True)
    parser.add_argument("--video", help="Source video for preview extraction")
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--category", default="ball")
    parser.add_argument("--limit", type=int, help="Cap number of labels (debug)")
    parser.add_argument("--no-frames", action="store_true",
                        help="Write manifest only; skip frame extraction")
    args = parser.parse_args()

    rows = load_ball_annotations(Path(args.coco), args.category)
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        print("no ball annotations found", file=sys.stderr)
        return 1

    camera_id = Path(args.video).name if args.video else "padeltracker100"
    for row in rows:
        row["camera_id"] = camera_id
    assign_splits(rows)

    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = 30.0
    original_width = rows[0].get("width")
    original_height = rows[0].get("height")
    scale_by_frame: dict[int, float] = {}

    if not args.no_frames:
        if not args.video:
            parser.error("--video is required unless --no-frames is set")
        import cv2
        frames_dir.mkdir(exist_ok=True)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            print(f"cannot open video {args.video}", file=sys.stderr)
            return 1
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        for index, row in enumerate(rows):
            cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            scale = args.width / frame.shape[1]
            preview = cv2.resize(
                frame, (args.width, int(frame.shape[0] * scale)))
            cv2.imwrite(str(frames_dir / f"frame_{row['frame']:06d}.jpg"), preview,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            scale_by_frame[row["frame"]] = scale
            if index and index % 500 == 0:
                print(f"extracted {index}/{len(rows)} frames", flush=True)
        cap.release()

    labels = []
    for row in rows:
        labels.append({
            "frame": row["frame"],
            "t": row["frame"] / fps,
            "image": f"frames/frame_{row['frame']:06d}.jpg",
            "image_scale": scale_by_frame.get(row["frame"]),
            "state": "visible",
            "center": [round(row["center"][0], 2), round(row["center"][1], 2)],
            "event_tags": [],
            "sequence_id": row["sequence_id"],
            "camera_id": row["camera_id"],
            "split": row["split"],
        })
    doc = {
        "schema_version": "1.0",
        "video": args.video,
        "fps": fps,
        "coordinate_space": "original_video_pixels",
        "preview_width": args.width,
        "original_video_width": original_width,
        "original_video_height": original_height,
        "source": "PadelTracker100 (Zenodo 14653706, CC-BY-4.0)",
        "labels": labels,
    }
    (out_dir / "labels.json").write_text(json.dumps(doc, indent=1))
    counts = {}
    for label in labels:
        counts[label["split"]] = counts.get(label["split"], 0) + 1
    print(f"wrote {len(labels)} labels -> {out_dir / 'labels.json'} splits={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
