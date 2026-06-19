#!/usr/bin/env python
"""Single-camera ball-candidate diagnostic overlay.

Draws every colour+motion candidate in yellow and the continuity-selected
candidate in red. This is for judging 2D detection quality before any multi-cam
triangulation or scoring is involved.
"""
import argparse
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import cv2

from ball_motion_color import ball_candidates


def parse_hsv(value):
    return [int(part.strip()) for part in value.split(",")]


def candidate_quality(candidate):
    radius_penalty = max(float(candidate.get("r", 4.0)) - 4.0, 0.0) * 0.08
    aspect_penalty = max(float(candidate.get("aspect_ratio", 1.0)) - 1.0, 0.0) * 0.4
    return (
        2.0 * float(candidate.get("circularity", 0.0))
        + float(candidate.get("fill_ratio", 0.0))
        - radius_penalty
        - aspect_penalty
    )


def choose_candidate(candidates, previous=None, continuity_weight=0.03):
    if not candidates:
        return None
    best = None
    best_score = None
    for candidate in candidates:
        score = candidate_quality(candidate)
        if previous is not None:
            dist = math.dist(
                (candidate["x"], candidate["y"]),
                (previous["x"], previous["y"]),
            )
            score -= continuity_weight * dist
        if best_score is None or score > best_score:
            best = candidate
            best_score = score
    return best


def read_neighbor_frames(cap, frame_no):
    frames = []
    for offset in (-1, 0, 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_no + offset, 0))
        ok, frame = cap.read()
        frames.append(frame if ok else None)
    return frames


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"could not open writer for {path}")


def draw_overlay(frame, candidates, selected, trail):
    for candidate in candidates:
        x, y, r = candidate["x"], candidate["y"], candidate.get("r", 4.0)
        cv2.circle(frame, (int(x), int(y)), max(int(r) + 8, 12), (0, 255, 255), 2)
    if selected is not None:
        trail.append((selected["x"], selected["y"]))
        if len(trail) > 20:
            trail.pop(0)
    for i, (x, y) in enumerate(trail):
        alpha = (i + 1) / max(len(trail), 1)
        cv2.circle(frame, (int(x), int(y)), max(2, int(3 + 5 * alpha)),
                   (0, int(160 * alpha), 255), -1)
    if selected is not None:
        x, y = selected["x"], selected["y"]
        cv2.circle(frame, (int(x), int(y)), 16, (0, 0, 255), 3)
        cv2.putText(frame, "selected 2D", (int(x) + 18, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--window", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--hsv-lo", default="22,50,110")
    parser.add_argument("--hsv-hi", default="48,255,255")
    parser.add_argument("--motion-thr", type=int, default=18)
    parser.add_argument("--min-circularity", type=float, default=0.35)
    parser.add_argument("--min-fill-ratio", type=float, default=0.30)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.2)
    parser.add_argument("--min-radius", type=float, default=2.5)
    parser.add_argument("--max-radius", type=float, default=18.0)
    parser.add_argument("--out", default="/tmp/single_cam_ball_overlay.mp4")
    args = parser.parse_args()

    video = args.video if os.path.isabs(args.video) else os.path.join(_ROOT, args.video)
    lo, hi = parse_hsv(args.hsv_lo), parse_hsv(args.hsv_hi)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / in_w
    out_size = (args.width, int(in_h * scale))
    writer = open_writer(args.out, fps, out_size)

    f0 = int(args.start * fps)
    f1 = int((args.start + args.window) * fps)
    previous = None
    trail = []
    frames = 0
    selected_frames = 0
    total_candidates = 0

    for frame_no in range(f0, f1):
        prev, cur, nxt = read_neighbor_frames(cap, frame_no)
        if cur is None:
            break
        candidates = []
        if prev is not None and nxt is not None:
            candidates = ball_candidates(
                prev, cur, nxt, lo, hi, 2.0, 400.0,
                motion_thr=args.motion_thr,
                min_circularity=args.min_circularity,
                min_fill_ratio=args.min_fill_ratio,
                max_aspect_ratio=args.max_aspect_ratio,
                min_radius=args.min_radius,
                max_radius=args.max_radius,
                return_metrics=True,
            )
        selected = choose_candidate(candidates, previous=previous)
        if selected is not None:
            previous = selected
            selected_frames += 1
        total_candidates += len(candidates)
        draw_overlay(cur, candidates, selected, trail)
        cv2.putText(cur, f"single-cam 2D candidates={len(candidates)} selected={selected is not None}",
                    (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(cur, f"t={frame_no / fps:.2f}s  yellow=all candidates red=selected",
                    (30, cur.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        writer.write(cv2.resize(cur, out_size))
        frames += 1
        if frames % 100 == 0:
            print(f"\r  frames={frames} selected={selected_frames}", end="", file=sys.stderr, flush=True)

    cap.release()
    writer.release()
    print(file=sys.stderr)
    print(f"wrote {args.out}")
    print(f"  frames: {frames}")
    print(f"  frames with selected candidate: {selected_frames}")
    print(f"  avg candidates/frame: {total_candidates / frames:.2f}" if frames else "  avg candidates/frame: 0")


if __name__ == "__main__":
    main()
