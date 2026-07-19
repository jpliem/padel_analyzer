#!/usr/bin/env python3
"""Extract mono audio with ffmpeg and emit impact candidates as JSON."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend", "src"))

from cv.audio_events import AudioImpulseDetector


def analyze_samples(samples: np.ndarray, video: str, sample_rate: int,
                    threshold_mad: float) -> dict:
    """Build an honest audio-evidence payload, including silent tracks."""
    samples = np.asarray(samples, dtype="<f4")
    if samples.size == 0 or float(np.max(np.abs(samples))) <= 1e-8:
        return {
            "video": video,
            "sample_rate": sample_rate,
            "events": [],
            "status": "silent_audio",
            "warning": "The audio stream is silent; contact fusion must use visual evidence.",
        }

    detector = AudioImpulseDetector(sample_rate, threshold_mad=threshold_mad)
    events = detector.detect(samples)
    return {
        "video": video,
        "sample_rate": sample_rate,
        "events": [event.__dict__ for event in events],
        "status": "ok",
        "warning": "Impact candidates are evidence, not classified padel contacts.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--threshold-mad", type=float, default=8.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    command = [
        "ffmpeg", "-v", "error", "-i", args.video, "-vn", "-ac", "1",
        "-ar", str(args.sample_rate), "-f", "f32le", "-",
    ]
    completed = subprocess.run(command, check=False, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    if completed.returncode != 0:
        payload = {
            "video": args.video, "sample_rate": args.sample_rate, "events": [],
            "status": "no_decodable_audio",
            "warning": "Audio is unavailable; contact fusion must use visual evidence.",
        }
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"no decodable audio; wrote {args.out}")
        return 0
    samples = np.frombuffer(completed.stdout, dtype="<f4")
    payload = analyze_samples(
        samples, args.video, args.sample_rate, args.threshold_mad
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(f"wrote {len(payload['events'])} candidates to {args.out}"
          f" (status={payload['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
