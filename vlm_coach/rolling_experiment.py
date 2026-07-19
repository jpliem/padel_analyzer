from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from .mlx_client import MlxVlmClient
from .schemas import model_dump
from .scoring import (
    ROLLING_PROMPT, RollingObservation, rolling_context, rolling_windows,
    reconcile_rolling_observation, sanitize_frame_evidence,
)
from .video import extract_storyboard, probe_video


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stateful chronological VLM scan for padel rally boundaries"
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--near-team", choices=("team_a", "team_b"), default="team_a")
    parser.add_argument("--window", type=float, default=12.0)
    parser.add_argument("--overlap", type=float, default=3.0)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", help="Optional path for machine-readable result JSON")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    media = probe_video(video)
    scan_end = min(media["duration"], args.start + args.duration) if args.duration else media["duration"]
    if scan_end <= args.start:
        raise ValueError("scan duration must include at least one second")
    near, far = args.near_team, "team_b" if args.near_team == "team_a" else "team_a"
    windows = rolling_windows(scan_end - args.start, args.window, args.overlap)
    windows = [type(item)(item.start + args.start, item.end + args.start) for item in windows]
    windows = windows[:args.limit or None]

    client = MlxVlmClient()
    previous = None
    results = []
    total_seconds = 0.0
    with tempfile.TemporaryDirectory(prefix="padel-rolling-") as temporary:
        for index, window in enumerate(windows):
            frames = extract_storyboard(
                video, window, Path(temporary) / f"window_{index:03d}",
                frame_count=max(2, args.frames), sampling="uniform",
                annotate_timeline=True,
            )
            timeline = ", ".join(
                f"frame {frame['index']}={frame['timestamp']:.2f}s" for frame in frames
            )
            prompt = ROLLING_PROMPT.format(
                near_team=near, far_team=far,
                previous_state=rolling_context(previous),
            ) + f"\nTHIS window timeline: {timeline}."
            started = time.perf_counter()
            observation = client.structured(
                args.model, prompt, RollingObservation,
                images=[frame["path"] for frame in frames],
            )
            elapsed = time.perf_counter() - started
            payload = sanitize_frame_evidence(model_dump(observation), len(frames))
            payload = reconcile_rolling_observation(previous, payload)
            previous = payload
            total_seconds += elapsed
            results.append({
                "window": vars(window), "frames": len(frames),
                "frame_timestamps": [frame["timestamp"] for frame in frames],
                "seconds": round(elapsed, 3), "analysis": payload,
            })

    output = {
        "video": str(video), "model": args.model,
        "configuration": {
            "window": args.window, "overlap": args.overlap, "frames": args.frames,
            "near": near, "far": far,
        },
        "total_vlm_seconds": round(total_seconds, 3), "windows": results,
    }
    rendered = json.dumps(output, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
