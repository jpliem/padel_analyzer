from __future__ import annotations

import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import cv2

from .mlx_client import MlxVlmClient
from .schemas import model_dump
from .scoring import (
    RollingObservation, reconcile_rolling_observation, sanitize_frame_evidence,
)
from .video import probe_video


NATIVE_VIDEO_PROMPT = """Inspect this padel video in chronological order. The fixed camera
is behind one baseline; lower/larger players are near and upper/smaller players
are far. Decide whether a rally is visible and whether active play visibly ends.
A missing tiny ball is not evidence of an ending. Do not name a winner or score.
For native video input, evidence_frames refers to sampled-frame order starting at
zero. Cite active_play_frames before later reset_frames for any claimed ending.
Return only schema JSON."""


def trim_video(source: Path, destination: Path, start: float, duration: float) -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError("Video could not be opened")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise ValueError("Temporary video clip could not be created")
    remaining = max(1, int(round(duration * fps)))
    for _ in range(remaining):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
    writer.release()
    capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Qwen native video ingestion")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--fps", type=float, default=1.5)
    args = parser.parse_args()

    source = Path(args.video).resolve()
    media = probe_video(source)
    duration = min(args.duration, media["duration"] - args.start)
    if duration <= 0:
        raise ValueError("Requested clip is outside the video")
    with tempfile.TemporaryDirectory(prefix="padel-native-video-") as temporary:
        clip = Path(temporary) / "clip.mp4"
        trim_video(source, clip, args.start, duration)
        client = MlxVlmClient()
        started = time.perf_counter()
        result = client.structured(
            args.model, NATIVE_VIDEO_PROMPT, RollingObservation,
            videos=[clip], video_fps=args.fps,
        )
        elapsed = time.perf_counter() - started
        expected_frames = max(1, math.ceil(duration * args.fps))
        analysis = sanitize_frame_evidence(model_dump(result), expected_frames)
        analysis = reconcile_rolling_observation(None, analysis)
    print(json.dumps({
        "provider": "mlx-native-video", "model": args.model,
        "start": args.start, "duration": duration, "sampling_fps": args.fps,
        "expected_sampled_frames": expected_frames,
        "inference_seconds": round(elapsed, 3), "analysis": analysis,
    }, indent=2))


if __name__ == "__main__":
    main()
