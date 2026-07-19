#!/usr/bin/env python3
"""Find candidate CalTennis windows with fast yellow-object motion.

This is a search tool, not a label generator. It ranks moments for visual
review so stationary balls and the player's lime racket do not get silently
accepted as the active ball.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ball_motion_color import ball_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=600.0)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--hsv-low", default="25,90,140")
    parser.add_argument("--hsv-high", default="42,255,255")
    parser.add_argument("--min-speed-px-s", type=float, default=120.0)
    parser.add_argument("--max-speed-px-s", type=float, default=5000.0)
    parser.add_argument("--top", type=int, default=40)
    parser.add_argument("--out", default="data/experiments/caltennis_activity_scan.json")
    args = parser.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = ROOT / video
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    low = [int(value) for value in args.hsv_low.split(",")]
    high = [int(value) for value in args.hsv_high.split(",")]

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 60.0
    step = max(1, round(fps / args.rate))
    start_frame = round(args.start * fps)
    end_frame = round((args.start + args.duration) * fps)
    previous_candidates = []
    previous_time = None
    moments = []
    # Decode sequentially. Repeated random frame seeks are extremely slow on
    # long-GOP H.264 video and made a five-minute scan take longer than real
    # time on some machines.
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame - 1))
    ok_previous, previous_frame = capture.read()
    ok_current, current_frame = capture.read()
    if not ok_previous or not ok_current:
        raise SystemExit("could not read scan start")
    frame_index = start_frame
    while frame_index <= end_frame:
        ok_next, next_frame = capture.read()
        if not ok_next:
            break
        if (frame_index - start_frame) % step:
            previous_frame, current_frame = current_frame, next_frame
            frame_index += 1
            continue
        triplet = [previous_frame, current_frame, next_frame]
        current = ball_candidates(
            *triplet, low, high, 2.0, 300.0, motion_thr=14,
            min_circularity=0.18, min_fill_ratio=0.18,
            max_aspect_ratio=3.0, min_radius=1.5, max_radius=14.0,
            return_metrics=True,
        )
        timestamp = frame_index / fps
        if previous_time is not None:
            dt = timestamp - previous_time
            for candidate in current:
                if not previous_candidates:
                    continue
                distances = [
                    float(np.hypot(candidate["x"] - prior["x"], candidate["y"] - prior["y"]))
                    for prior in previous_candidates
                ]
                distance = min(distances)
                speed = distance / dt
                if args.min_speed_px_s <= speed <= args.max_speed_px_s:
                    moments.append({
                        "video_time": timestamp,
                        "frame": frame_index,
                        "pixel": [candidate["x"], candidate["y"]],
                        "speed_px_s": speed,
                        "candidate": candidate,
                    })
        previous_candidates = current
        previous_time = timestamp
        previous_frame, current_frame = current_frame, next_frame
        frame_index += 1
    capture.release()
    moments.sort(key=lambda item: item["speed_px_s"], reverse=True)
    result = {
        "video": str(video),
        "fps": fps,
        "scan_window": [args.start, args.start + args.duration],
        "sample_rate_hz": args.rate,
        "moments": moments[:args.top],
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
