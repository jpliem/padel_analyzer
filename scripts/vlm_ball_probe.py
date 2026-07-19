#!/usr/bin/env python
"""Probe a VLM point-grounding model for padel ball localization.

This is intentionally a sparse, offline experiment. It asks Molmo2-style video
pointing models to point to the padel ball, parses the returned coordinates,
and optionally scores them against PADELVIC synthetic ground truth.
"""
import argparse
import json
import math
import os
import re
import sys
from statistics import median


COORD_REGEX = re.compile(r"<(?:points|tracks).*? coords=\"([0-9\t:;, .]+)\"/?>")
FRAME_REGEX = re.compile(r"^\s*([0-9.]+)\s*:?\s*(.*)$")
POINTS_REGEX = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def extract_video_points(text, image_w, image_h):
    """Extract Molmo2 video-pointing output as pixel coordinates.

    Molmo2 emits coordinates scaled to 0..1000 inside text such as:
    ``<points coords="0: 1 500 250; 1: 1 700 300"/>``.
    """
    points = []
    for coord in COORD_REGEX.finditer(text):
        for chunk in coord.group(1).replace("\t", ";").split(";"):
            point_group = FRAME_REGEX.match(chunk)
            if not point_group:
                continue
            frame_id = float(point_group.group(1))
            for match in POINTS_REGEX.finditer(point_group.group(2)):
                point_id, x_raw, y_raw = match.groups()
                x = float(x_raw) / 1000.0 * image_w
                y = float(y_raw) / 1000.0 * image_h
                if 0 <= x <= image_w and 0 <= y <= image_h:
                    points.append({
                        "frame": frame_id,
                        "id": point_id,
                        "x": round(x, 2),
                        "y": round(y, 2),
                    })
    return points


def load_gt(csv_path):
    gt = []
    with open(csv_path) as f:
        for line in f:
            parts = line.strip().replace(",", ";").split(";")
            if len(parts) < 3:
                gt.append(None)
                continue
            try:
                gt.append((float(parts[1]), float(parts[2])))
            except ValueError:
                gt.append(None)
    return gt


def load_label_gt(labels_path):
    """Load reviewed v1 ball labels into a frame-indexed sparse list."""
    with open(labels_path) as handle:
        doc = json.load(handle)
    reviewed = [item for item in doc.get("labels", [])
                if item.get("state") in ("visible", "blurred") and item.get("center")]
    size = max((int(item["frame"]) for item in reviewed), default=-1) + 1
    gt = [None] * size
    for item in reviewed:
        gt[int(item["frame"])] = tuple(float(v) for v in item["center"])
    return gt


def summarize_against_gt(points, gt, threshold_px=50):
    errors = []
    matched = []
    for point in points:
        frame = int(round(point["frame"]))
        if frame < 0 or frame >= len(gt) or gt[frame] is None:
            continue
        truth = gt[frame]
        err = math.dist((point["x"], point["y"]), truth)
        errors.append(err)
        matched.append({
            "frame": frame,
            "pred": [point["x"], point["y"]],
            "gt": [truth[0], truth[1]],
            "error_px": round(err, 2),
        })

    return {
        "matched_points": len(errors),
        "mean_error_px": round(sum(errors) / len(errors), 2) if errors else None,
        "median_error_px": round(median(errors), 2) if errors else None,
        "pck": round(sum(1 for e in errors if e <= threshold_px) / len(errors), 4)
               if errors else None,
        "threshold_px": threshold_px,
        "matches": matched,
    }


def run_molmo2(model_id, video, prompt, max_new_tokens):
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        from molmo_utils import process_vision_info
    except ImportError as exc:
        raise SystemExit(
            "Missing VLM dependencies. Install the model-card stack first:\n"
            "  pip install transformers==4.57.1 accelerate einops torchvision decord2 molmo_utils\n"
            f"Original import error: {exc}"
        )

    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video", "video": video},
        ],
    }]
    processor = AutoProcessor.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", device_map="auto")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, trust_remote_code=True, dtype="auto", device_map="auto")

    _, videos, video_kwargs = process_vision_info(messages)
    videos, video_metadatas = zip(*videos)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        videos=list(videos),
        video_metadata=list(video_metadatas),
        text=text,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_tokens = generated_ids[0, inputs["input_ids"].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    metadata = video_metadatas[0]
    return generated_text, metadata["width"], metadata["height"]


def main():
    parser = argparse.ArgumentParser(description="Try Molmo2/VLM ball pointing on a video.")
    parser.add_argument("--video", help="Video file/URL. Required unless --raw-text is used.")
    parser.add_argument("--model-id", default="allenai/Molmo2-VideoPoint-4B")
    parser.add_argument("--prompt", default="Point to the padel ball in each visible frame.")
    parser.add_argument("--labels", help="Optional reviewed v1 ball labels.json for scoring")
    parser.add_argument("--threshold-px", type=float, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--raw-text", help="Parse an existing model response instead of loading a VLM")
    parser.add_argument("--image-width", type=int, help="Required with --raw-text")
    parser.add_argument("--image-height", type=int, help="Required with --raw-text")
    parser.add_argument("--out", default="/tmp/vlm_ball_probe.json")
    args = parser.parse_args()

    if args.raw_text:
        if not args.image_width or not args.image_height:
            parser.error("--raw-text requires --image-width and --image-height")
        with open(args.raw_text) as f:
            generated_text = f.read()
        image_w, image_h = args.image_width, args.image_height
    else:
        if not args.video:
            parser.error("--video is required unless --raw-text is used")
        generated_text, image_w, image_h = run_molmo2(
            args.model_id, args.video, args.prompt, args.max_new_tokens)

    points = extract_video_points(generated_text, image_w, image_h)
    result = {
        "model_id": args.model_id,
        "video": args.video,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "points": points,
    }
    if args.labels:
        result["summary"] = summarize_against_gt(
            points, load_label_gt(args.labels), threshold_px=args.threshold_px)

    out = args.out if os.path.isabs(args.out) else os.path.abspath(args.out)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    if "summary" in result:
        print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
