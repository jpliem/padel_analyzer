from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from .mlx_client import MlxVlmClient
from .schemas import model_dump
from .scoring import FRAME_STATE_PROMPT, FrameStateObservation, derive_frame_state_window
from .video import Segment, extract_storyboard, probe_video, storyboard_panels


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-frame padel state classification")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--representation", choices=("images", "panels"), default="images")
    parser.add_argument("--output")
    args = parser.parse_args()
    video = Path(args.video).resolve()
    media = probe_video(video)
    end = min(media["duration"], args.start + args.duration)
    with tempfile.TemporaryDirectory(prefix="padel-frame-state-") as temporary:
        frames = extract_storyboard(
            video, Segment(args.start, end), Path(temporary),
            frame_count=args.frames, annotate_timeline=True,
        )
        visual_paths = [item["path"] for item in frames]
        if args.representation == "panels":
            visual_paths = storyboard_panels(frames, Path(temporary) / "panels")
        prompt = FRAME_STATE_PROMPT + f"\nThere are exactly {len(frames)} frames, indexed 0 through {len(frames)-1}."
        if args.representation == "panels":
            prompt += " Frames are packed into 2x2 panels in left-to-right, top-to-bottom order; use the visible FRAME labels."
        client = MlxVlmClient()
        started = time.perf_counter()
        result = client.structured(
            args.model, prompt, FrameStateObservation,
            images=visual_paths, max_tokens=900,
        )
        elapsed = time.perf_counter() - started
    raw = model_dump(result)
    output = {
        "video": str(video), "model": args.model,
        "window": {"start": args.start, "end": end},
        "representation": args.representation,
        "visual_inputs": len(visual_paths),
        "frame_timestamps": [item["timestamp"] for item in frames],
        "inference_seconds": round(elapsed, 3), "raw": raw,
        "derived": derive_frame_state_window(raw, len(frames)),
    }
    rendered = json.dumps(output, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
