from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from .video import Segment, extract_storyboard, probe_video


DEFAULT_MODEL = "starkdmi/Molmo2-4B-4bit"
COORD_BLOCK = re.compile(r'<(?:points|tracks).*?coords="([0-9\t:;, .]+)"/?>')
POINT = re.compile(r"([0-9]+) ([0-9]{3,4}) ([0-9]{3,4})")


def parse_molmo_points(text: str, sizes: list[tuple[int, int]]) -> list[dict]:
    """Decode Molmo2's 0-1000 multi-image point coordinates."""
    points: list[dict] = []
    for block in COORD_BLOCK.finditer(text):
        for raw_group in re.split(r"[\t:;,]", block.group(1)):
            values = raw_group.strip().split(maxsplit=1)
            if len(values) != 2:
                continue
            frame_number = int(float(values[0]))
            frame_index = frame_number - 1
            if not 0 <= frame_index < len(sizes):
                continue
            width, height = sizes[frame_index]
            for match in POINT.finditer(values[1]):
                object_id, x, y = map(int, match.groups())
                points.append({
                    "frame": frame_index,
                    "object_id": object_id,
                    "x": round(x / 1000 * width, 1),
                    "y": round(y / 1000 * height, 1),
                    "x_normalized": x / 1000,
                    "y_normalized": y / 1000,
                })
    return points


def render_contact_sheet(frames: list[dict], points: list[dict], output: Path) -> None:
    annotated = []
    grouped: dict[int, list[dict]] = {}
    for point in points:
        grouped.setdefault(point["frame"], []).append(point)
    for frame in frames:
        image = cv2.imread(str(frame["path"]))
        for point in grouped.get(frame["index"], []):
            center = (round(point["x"]), round(point["y"]))
            cv2.circle(image, center, 12, (0, 255, 255), 3)
            cv2.putText(image, str(point["object_id"]), (center[0] + 14, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f"frame {frame['index']}  {frame['timestamp']:.1f}s", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        annotated.append(image)
    if not annotated:
        return
    height = max(image.shape[0] for image in annotated)
    width = max(image.shape[1] for image in annotated)
    tiles = [cv2.resize(image, (width, height)) for image in annotated]
    columns = 2
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index:index + columns])
            for index in range(0, len(tiles), columns)]
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Molmo2 player grounding")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--output", default="data/benchmarks/molmo2_players.jpg")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    media = probe_video(video)
    end = min(media["duration"], args.start + args.duration)
    if end <= args.start:
        raise SystemExit("The requested window is outside the recording")

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template

    with tempfile.TemporaryDirectory(prefix="padel-molmo-benchmark-") as temporary:
        frames = extract_storyboard(
            video, Segment(args.start, end), Path(temporary), frame_count=args.frames,
        )
        model, processor = load(args.model)
        config = getattr(model, "config", None)
        prompt = (
            "Point to the center of every visible padel player in every image. "
            "Keep the same object ID for the same player across images. Return native "
            "Molmo points only; do not describe the scene."
        )
        formatted = apply_chat_template(
            processor, config, prompt, num_images=len(frames),
            add_generation_prompt=True, enable_thinking=False,
        )
        started = time.perf_counter()
        result = generate(
            model, processor, formatted,
            image=[str(frame["path"]) for frame in frames],
            max_tokens=500, temperature=0.0, verbose=False,
        )
        elapsed = time.perf_counter() - started
        sizes = []
        for frame in frames:
            image = cv2.imread(str(frame["path"]))
            sizes.append((image.shape[1], image.shape[0]))
        points = parse_molmo_points(result.text, sizes)
        output = Path(args.output).resolve()
        render_contact_sheet(frames, points, output)

    print(json.dumps({
        "provider": "mlx",
        "model": args.model,
        "window_seconds": round(end - args.start, 3),
        "frames": len(frames),
        "inference_seconds": round(elapsed, 3),
        "points_found": len(points),
        "points": points,
        "raw_output": result.text,
        "annotated_output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
