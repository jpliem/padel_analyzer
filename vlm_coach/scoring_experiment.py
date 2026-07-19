from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from .mlx_client import MlxVlmClient
from .scoring import (
    SCOUT_PROMPT, VERDICT_PROMPT, RallyScout, RallyVerdict,
    add_scoring_context, fixed_windows, merge_scout_windows, sanitize_frame_evidence,
)
from .schemas import model_dump
from .video import Segment, detect_segments, extract_storyboard, probe_video


def _storyboard(client, model: str, video: Path, segment: Segment, directory: Path,
                prompt: str, output_type, frame_count: int) -> tuple[dict, float, list[dict]]:
    frames = extract_storyboard(
        video, segment, directory, frame_count=frame_count, sampling="scoring",
    )
    timeline = ", ".join(
        f"frame {item['index']}={item['timestamp']:.1f}s" for item in frames
    )
    started = time.perf_counter()
    result = client.structured(
        model, f"{prompt}\nStoryboard timeline: {timeline}.", output_type,
        images=[item["path"] for item in frames],
    )
    payload = sanitize_frame_evidence(model_dump(result), len(frames))
    return payload, time.perf_counter() - started, frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare padel scoring architectures")
    parser.add_argument("--video", required=True)
    parser.add_argument(
        "--mode", choices=("opencv", "vlm", "hybrid", "multi", "cascade"),
        required=True,
    )
    parser.add_argument("--scout-model", default="qwen3.5:0.8b")
    parser.add_argument("--judge-model", default="qwen3.5:2b")
    parser.add_argument("--near-team", choices=("team_a", "team_b"), default="team_a")
    parser.add_argument("--score-before", default="0-0")
    parser.add_argument("--window", type=float, default=6.0)
    parser.add_argument("--stride", type=float, default=3.0)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit candidates/windows for a cheap benchmark; 0 means all")
    parser.add_argument("--candidate-offset", type=int, default=0,
                        help="Skip this many OpenCV candidates before applying --limit")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    media = probe_video(video)
    far_team = "team_b" if args.near_team == "team_a" else "team_a"
    raw_cv_candidates = detect_segments(video)
    cv_candidates = add_scoring_context(raw_cv_candidates, media["duration"])
    result = {
        "mode": args.mode, "video": str(video), "media": media,
        "camera": {"near": args.near_team, "far": far_team},
        "opencv_candidates": [vars(item) for item in raw_cv_candidates],
        "scouts": [], "candidates": [], "verdicts": [], "vlm_seconds": 0.0,
    }

    if args.mode == "opencv":
        candidates = cv_candidates[args.candidate_offset:]
        candidates = candidates[:args.limit or None]
        result["candidates"] = [vars(item) for item in candidates]
        result["note"] = "Motion can propose rally windows but cannot safely award points."
        print(json.dumps(result, indent=2))
        return

    client = MlxVlmClient()
    with tempfile.TemporaryDirectory(prefix="padel-scoring-experiment-") as temporary:
        root = Path(temporary)
        if args.mode in ("vlm", "multi"):
            windows = fixed_windows(media["duration"], args.window, args.stride)
            if args.limit:
                windows = windows[:args.limit]
            previous_phase = "waiting"
            scout_items = []
            scout_model = args.judge_model if args.mode == "vlm" else args.scout_model
            for index, window in enumerate(windows):
                prompt = SCOUT_PROMPT.format(
                    near_team=args.near_team, far_team=far_team,
                    previous_phase=previous_phase,
                )
                scout, elapsed, frames = _storyboard(
                    client, scout_model, video, window, root / f"scout_{index:03d}",
                    prompt, RallyScout, max(4, min(args.frames, 12)),
                )
                previous_phase = scout["phase_at_end"]
                result["vlm_seconds"] += elapsed
                result["scouts"].append({
                    "window": vars(window), "model": scout_model,
                    "seconds": round(elapsed, 3), "analysis": scout,
                })
                scout_items.append((window, RallyScout(**scout)))
            candidates = merge_scout_windows(scout_items)
            candidates = [Segment(item.start, min(media["duration"], item.end))
                          for item in candidates]
        elif args.mode == "cascade":
            source = cv_candidates[args.candidate_offset:]
            source = source[:args.limit or None]
            candidates = []
            previous_phase = "waiting"
            for index, window in enumerate(source):
                prompt = SCOUT_PROMPT.format(
                    near_team=args.near_team, far_team=far_team,
                    previous_phase=previous_phase,
                )
                scout, elapsed, frames = _storyboard(
                    client, args.scout_model, video, window,
                    root / f"scout_{index:03d}", prompt, RallyScout,
                    max(4, min(args.frames, 10)),
                )
                previous_phase = scout["phase_at_end"]
                result["vlm_seconds"] += elapsed
                result["scouts"].append({
                    "window": vars(window), "model": args.scout_model,
                    "seconds": round(elapsed, 3), "analysis": scout,
                })
                if (scout["rally_visible"] or scout["serve_candidate"]
                        or scout["point_end_candidate"]):
                    candidates.append(window)
        else:
            candidates = cv_candidates[args.candidate_offset:]
            candidates = candidates[:args.limit or None]

        result["candidates"] = [vars(item) for item in candidates]
        for index, candidate in enumerate(candidates):
            related = [item for item in result["scouts"]
                       if item["window"]["end"] >= candidate.start
                       and item["window"]["start"] <= candidate.end]
            evidence = json.dumps([item["analysis"] for item in related], separators=(",", ":"))
            prompt = VERDICT_PROMPT.format(
                near_team=args.near_team, far_team=far_team,
                score_before=args.score_before, scout_evidence=evidence or "none",
            )
            verdict, elapsed, frames = _storyboard(
                client, args.judge_model, video, candidate,
                root / f"verdict_{index:03d}", prompt, RallyVerdict, args.frames,
            )
            result["vlm_seconds"] += elapsed
            result["verdicts"].append({
                "candidate": vars(candidate), "model": args.judge_model,
                "seconds": round(elapsed, 3), "analysis": verdict,
            })

    result["vlm_seconds"] = round(result["vlm_seconds"], 3)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
