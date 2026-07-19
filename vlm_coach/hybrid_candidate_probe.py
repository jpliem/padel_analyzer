from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio_probe import quiet_intervals, read_audio_clip, spectral_impulses
from .scoring import gap_contexts
from .video import detect_segments, probe_video


def overlap_seconds(first: tuple[float, float], second: tuple[float, float]) -> float:
    return max(0.0, min(first[1], second[1]) - max(first[0], second[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse OpenCV gaps with audio quietness")
    parser.add_argument("--video", required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--minimum-audio-quiet", type=float, default=2.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    video = Path(args.video).resolve()
    media = probe_video(video)
    gaps = gap_contexts(detect_segments(video), media["duration"])
    gaps = gaps[args.offset:][:args.limit or None]
    results = []
    for gap, context in gaps:
        sample_rate, samples = read_audio_clip(video, context.start, context.duration)
        impulses = spectral_impulses(samples, sample_rate)
        relative_quiet = quiet_intervals(
            impulses["times"], context.duration, args.minimum_audio_quiet
        )
        absolute_quiet = [
            [round(context.start + start, 3), round(context.start + end, 3)]
            for start, end in relative_quiet
        ]
        overlaps = [
            round(overlap_seconds((gap.start, gap.end), tuple(interval)), 3)
            for interval in absolute_quiet
        ]
        best_overlap = max(overlaps, default=0.0)
        results.append({
            "gap": vars(gap), "context": vars(context),
            "audio_quiet_intervals": absolute_quiet,
            "best_gap_overlap_seconds": best_overlap,
            "audio_supports_boundary_review": best_overlap >= 1.5,
            "note": "Candidate gate only; quiet audio cannot award a point.",
        })
    rendered = json.dumps({
        "video": str(video), "minimum_audio_quiet": args.minimum_audio_quiet,
        "candidates": results,
    }, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
