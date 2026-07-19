from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from .mlx_client import MlxVlmClient
from .ollama_client import OllamaClient
from .pipeline import RALLY_PROMPT
from .schemas import RallyAnalysis, model_dump
from .video import Segment, extract_storyboard, probe_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark one real padel storyboard")
    parser.add_argument("--video", required=True)
    parser.add_argument("--provider", choices=("mlx", "ollama"), default="mlx")
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()

    video = Path(args.video).resolve()
    media = probe_video(video)
    end = min(media["duration"], args.start + args.duration)
    if end <= args.start:
        raise SystemExit("The requested window is outside the recording")
    client = MlxVlmClient() if args.provider == "mlx" else OllamaClient()
    with tempfile.TemporaryDirectory(prefix="padel-vlm-benchmark-") as temporary:
        frames = extract_storyboard(
            video, Segment(args.start, end), Path(temporary), frame_count=args.frames,
        )
        timeline = ", ".join(
            f"frame {item['index']}={item['timestamp']:.1f}s" for item in frames
        )
        prompt = RALLY_PROMPT.format(team_a="Team A", team_b="Team B")
        prompt += f"\nStoryboard timeline: {timeline}."
        started = time.perf_counter()
        result = client.structured(
            args.model, prompt, RallyAnalysis,
            images=[item["path"] for item in frames],
        )
        elapsed = time.perf_counter() - started
    print(json.dumps({
        "provider": args.provider,
        "model": args.model,
        "video": str(video),
        "window_seconds": round(end - args.start, 3),
        "frames": len(frames),
        "inference_seconds": round(elapsed, 3),
        "realtime_factor": round(elapsed / (end - args.start), 3),
        "analysis": model_dump(result),
    }, indent=2))


if __name__ == "__main__":
    main()

