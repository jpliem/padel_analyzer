#!/usr/bin/env python
"""Side-by-side synchronized camera diagnostic for 2D ball candidates."""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))

import cv2
import numpy as np

from annotate_single_cam_ball import (
    choose_candidate,
    draw_overlay,
    open_writer,
    parse_hsv,
    read_neighbor_frames,
)
from ball_motion_color import ball_candidates


def frame_for_time(t, fps, offset_seconds=0.0):
    return int(round((t - offset_seconds) * fps))


def stack_views(left, right, height=540):
    lw = int(left.shape[1] * height / left.shape[0])
    rw = int(right.shape[1] * height / right.shape[0])
    left_resized = cv2.resize(left, (lw, height))
    right_resized = cv2.resize(right, (rw, height))
    return np.hstack([left_resized, right_resized])


def resolve(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def annotate_camera_frame(cap, frame_no, lo, hi, previous, trail, label, args):
    prev, cur, nxt = read_neighbor_frames(cap, frame_no)
    if cur is None:
        return None, previous, 0, False
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
    draw_overlay(cur, candidates, selected, trail)
    cv2.putText(cur, f"{label} candidates={len(candidates)} selected={selected is not None}",
                (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return cur, previous, len(candidates), selected is not None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-a", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    parser.add_argument("--video-b", default="data/datasets/padelvic/cameras/gopro.mp4")
    parser.add_argument("--label-a", default="panasonic")
    parser.add_argument("--label-b", default="gopro")
    parser.add_argument("--sync", default="/tmp/sync_pana_gopro.json")
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--window", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=25.0)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--hsv-lo", default="22,50,110")
    parser.add_argument("--hsv-hi", default="48,255,255")
    parser.add_argument("--motion-thr", type=int, default=18)
    parser.add_argument("--min-circularity", type=float, default=0.35)
    parser.add_argument("--min-fill-ratio", type=float, default=0.30)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.2)
    parser.add_argument("--min-radius", type=float, default=2.5)
    parser.add_argument("--max-radius", type=float, default=18.0)
    parser.add_argument("--out", default="/tmp/multiview_panasonic_gopro_ball_overlay.mp4")
    args = parser.parse_args()

    video_a = resolve(args.video_a)
    video_b = resolve(args.video_b)
    cap_a = cv2.VideoCapture(video_a)
    cap_b = cv2.VideoCapture(video_b)
    fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 50.0
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 50.0
    offset = 0.0
    if args.sync and os.path.exists(args.sync):
        offset = json.load(open(args.sync)).get("offset_seconds", 0.0)

    lo, hi = parse_hsv(args.hsv_lo), parse_hsv(args.hsv_hi)
    writer = None
    previous_a = None
    previous_b = None
    trail_a = []
    trail_b = []
    frames = 0
    selected_a = 0
    selected_b = 0
    total_a = 0
    total_b = 0
    n = int(args.window * args.rate)

    for i in range(n):
        t = args.start + i / args.rate
        fa = frame_for_time(t, fps_a, 0.0)
        fb = frame_for_time(t, fps_b, offset)
        view_a, previous_a, count_a, hit_a = annotate_camera_frame(
            cap_a, fa, lo, hi, previous_a, trail_a, args.label_a, args)
        view_b, previous_b, count_b, hit_b = annotate_camera_frame(
            cap_b, fb, lo, hi, previous_b, trail_b, args.label_b, args)
        if view_a is None or view_b is None:
            break
        combined = stack_views(view_a, view_b, height=args.height)
        cv2.putText(combined, f"t={t:.2f}s yellow=all candidates red=selected",
                    (30, combined.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (255, 255, 255), 2)
        if writer is None:
            writer = open_writer(args.out, args.rate, (combined.shape[1], combined.shape[0]))
        writer.write(combined)
        frames += 1
        selected_a += int(hit_a)
        selected_b += int(hit_b)
        total_a += count_a
        total_b += count_b
        if frames % 100 == 0:
            print(f"\r  frames={frames}", end="", file=sys.stderr, flush=True)

    cap_a.release()
    cap_b.release()
    if writer is not None:
        writer.release()
    print(file=sys.stderr)
    print(f"wrote {args.out}")
    print(f"  frames: {frames}")
    print(f"  {args.label_a}: selected {selected_a}, avg candidates/frame {total_a / frames:.2f}" if frames else "")
    print(f"  {args.label_b}: selected {selected_b}, avg candidates/frame {total_b / frames:.2f}" if frames else "")


if __name__ == "__main__":
    main()
