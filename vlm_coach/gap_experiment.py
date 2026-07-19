from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

from .mlx_client import MlxVlmClient
from .scoring import (
    GAP_PROMPT, VERDICT_PROMPT, GapDecision, RallyVerdict, add_scoring_context,
    gap_contexts, merge_motion_bursts, sanitize_frame_evidence,
)
from .schemas import model_dump
from .video import detect_segments, extract_storyboard, probe_video, storyboard_panels


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify low-motion padel gaps")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", default="qwen3.5:0.8b")
    parser.add_argument("--near-team", choices=("team_a", "team_b"), default="team_a")
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--representation", choices=("images", "panels"), default="images")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--judge-model", default="",
                        help="Optional second model to adjudicate merged rallies")
    parser.add_argument("--judge-frames", type=int, default=12)
    parser.add_argument("--score-before", default="0-0")
    parser.add_argument("--output", default="", help="Optional JSON result path")
    args = parser.parse_args()

    video = Path(args.video).resolve()
    media = probe_video(video)
    bursts = detect_segments(video)
    all_gaps = gap_contexts(bursts, media["duration"])
    if args.judge_model and (args.offset != 0 or args.limit != 0):
        raise SystemExit("--judge-model requires --offset 0 --limit 0 to cover every gap")
    gaps = all_gaps[args.offset:]
    gaps = gaps[:args.limit or None]
    far_team = "team_b" if args.near_team == "team_a" else "team_a"
    client = MlxVlmClient()
    output = []
    decisions = []
    verdicts = []
    merged_candidates = []
    with tempfile.TemporaryDirectory(prefix="padel-gap-experiment-") as temporary:
        for index, (gap, context) in enumerate(gaps):
            frames = extract_storyboard(
                video, context, Path(temporary) / f"gap_{index:03d}",
                frame_count=args.frames, sampling="gap", annotate_timeline=True,
            )
            visual_paths = [item["path"] for item in frames]
            if args.representation == "panels":
                visual_paths = storyboard_panels(
                    frames, Path(temporary) / f"gap_{index:03d}_panels"
                )
            prompt = GAP_PROMPT.format(near_team=args.near_team, far_team=far_team)
            prompt += f"\nThere are exactly {len(frames)} images, numbered 0 through {len(frames)-1}."
            if args.representation == "panels":
                prompt += " Frames are packed into 2x2 panels left-to-right, top-to-bottom; use visible FRAME labels."
            started = time.perf_counter()
            decision = client.structured(
                args.model, prompt, GapDecision,
                images=visual_paths,
            )
            elapsed = time.perf_counter() - started
            payload = sanitize_frame_evidence(
                model_dump(decision), len(frames), strict=False,
            )
            decision = GapDecision(**payload)
            decisions.append(decision)
            output.append({
                "gap": vars(gap), "context": vars(context),
                "representation": args.representation,
                "visual_inputs": len(visual_paths),
                "seconds": round(elapsed, 3), "analysis": payload,
            })
        if args.offset == 0 and len(decisions) == len(all_gaps):
            merged = merge_motion_bursts(bursts, decisions)
            merged = add_scoring_context(merged, media["duration"])
            merged_candidates = [vars(item) for item in merged]
            for index, candidate in enumerate(merged if args.judge_model else []):
                frames = extract_storyboard(
                    video, candidate, Path(temporary) / f"verdict_{index:03d}",
                    frame_count=args.judge_frames, sampling="scoring",
                )
                prompt = VERDICT_PROMPT.format(
                    near_team=args.near_team, far_team=far_team,
                    score_before=args.score_before,
                    scout_evidence=json.dumps(output, separators=(",", ":")),
                )
                prompt += f"\nThere are exactly {len(frames)} images, numbered 0 through {len(frames)-1}."
                started = time.perf_counter()
                verdict = client.structured(
                    args.judge_model, prompt, RallyVerdict,
                    images=[item["path"] for item in frames],
                )
                elapsed = time.perf_counter() - started
                payload = sanitize_frame_evidence(model_dump(verdict), len(frames))
                verdicts.append({
                    "candidate": vars(candidate), "seconds": round(elapsed, 3),
                    "analysis": payload,
                })
    result = {
        "model": args.model, "video": str(video),
        "representation": args.representation,
        "motion_bursts": [vars(item) for item in bursts], "gaps": output,
        "merged_candidates": merged_candidates,
        "judge_model": args.judge_model or None, "verdicts": verdicts,
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        destination = Path(args.output).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
