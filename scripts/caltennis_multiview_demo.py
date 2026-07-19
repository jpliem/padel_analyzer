#!/usr/bin/env python3
"""Build a small CalTennis two-view ball/triangulation engineering proof.

The output is deliberately marked provisional: cross-view colour, motion, and
geometry produce strong label suggestions, but they are not padel ground truth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ball_motion_color import ball_candidates
from cv.caltennis import load_session
from cv.triangulation import reprojection_errors, triangulate


def read_triplet(capture, frame_index):
    frames = []
    for index in (frame_index - 1, frame_index, frame_index + 1):
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
        ok, frame = capture.read()
        frames.append(frame if ok else None)
    return frames


def tracknet_candidate(detector, frames):
    """Run a temporal TrackNet checkpoint on one contiguous frame triplet."""
    detector.reset()
    detection = None
    for offset, frame in enumerate(frames):
        detection = detector.detect(frame, offset)
    if detection is None:
        return []
    x1, y1, x2, y2 = detection
    return [{
        "x": (float(x1) + float(x2)) / 2.0,
        "y": (float(y1) + float(y2)) / 2.0,
        "r": max(2.0, (float(x2) - float(x1)) / 2.0),
        "source": "tracknet",
    }]


def choose_pair(candidates_a, candidates_b, projection_a, projection_b,
                max_reprojection_px, previous=None):
    best = None
    for candidate_a in candidates_a:
        for candidate_b in candidates_b:
            pixel_a = (candidate_a["x"], candidate_a["y"])
            pixel_b = (candidate_b["x"], candidate_b["y"])
            observations = [(projection_a, pixel_a), (projection_b, pixel_b)]
            point = triangulate(observations)
            if point is None or not np.all(np.isfinite(point)):
                continue
            errors = reprojection_errors(point, observations)
            max_error = max(errors)
            # CalTennis world Z is vertical. Allow a small ground/calibration
            # tolerance and realistic tennis-ball heights.
            if max_error > max_reprojection_px or not -0.5 <= point[2] <= 6.0:
                continue
            if np.linalg.norm(point[:2]) > 60.0:
                continue
            continuity = 0.0 if previous is None else float(np.linalg.norm(point - previous))
            score = max_error + 0.08 * continuity
            if best is None or score < best["score"]:
                best = {
                    "point": point,
                    "pixel_a": pixel_a,
                    "pixel_b": pixel_b,
                    "errors": errors,
                    "score": score,
                }
    return best


def draw_candidate_frame(frame, candidates, selected, camera_name, session_time):
    output = frame.copy()
    for candidate in candidates:
        cv2.circle(
            output, (round(candidate["x"]), round(candidate["y"])),
            max(8, round(candidate["r"]) + 4), (0, 200, 255), 1,
        )
    if selected is not None:
        cv2.circle(output, (round(selected[0]), round(selected[1])), 14, (0, 0, 255), 3)
    cv2.putText(
        output, f"{camera_name} session={session_time:.3f}s",
        (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    return output


def write_pair_preview(path, frame_a, frame_b, candidates_a, candidates_b, match,
                       names, session_time):
    selected_a = match["pixel_a"] if match else None
    selected_b = match["pixel_b"] if match else None
    images = [
        draw_candidate_frame(frame_a, candidates_a, selected_a, names[0], session_time),
        draw_candidate_frame(frame_b, candidates_b, selected_b, names[1], session_time),
    ]
    target_height = 540
    images = [cv2.resize(image, (round(image.shape[1] * target_height / image.shape[0]), target_height))
              for image in images]
    cv2.imwrite(str(path), np.hstack(images))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="data/datasets/caltennis")
    parser.add_argument("--session", default="01_23_2026_17_00_court2")
    parser.add_argument("--start", type=float, default=90.0,
                        help="shared session seconds from the earliest camera start")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--rate", type=float, default=15.0)
    parser.add_argument("--max-reprojection", type=float, default=4.0)
    parser.add_argument("--hsv-low", default="18,55,100")
    parser.add_argument("--hsv-high", default="48,255,255")
    parser.add_argument("--detector", choices=("color", "tracknet", "hybrid"),
                        default="hybrid")
    parser.add_argument("--tracknet-model", default="backend/models/tracknet_tennis.pt")
    parser.add_argument("--tracknet-threshold", type=float, default=0.5)
    parser.add_argument("--sync-search-frames", type=int, default=2,
                        help="search +/- frames in camera B for residual sub-frame sync error")
    parser.add_argument(
        "--review-status",
        choices=("provisional_cross_view_geometry", "visually_verified"),
        default="provisional_cross_view_geometry",
        help="only use visually_verified after inspecting every emitted preview",
    )
    parser.add_argument("--out", default="data/experiments/caltennis_multiview_demo")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    output = Path(args.out)
    if not output.is_absolute():
        output = ROOT / output
    previews = output / "previews"
    label_frames = output / "frames"
    for directory in (previews, label_frames):
        directory.mkdir(parents=True, exist_ok=True)
        for stale_preview in directory.glob("*.jpg"):
            stale_preview.unlink()

    session = load_session(dataset_root, args.session)
    if len(session.streams) != 2:
        raise SystemExit(f"demo expects exactly two streams; found {len(session.streams)}")
    stream_a, stream_b = session.streams
    overlap_start, overlap_end = session.overlap_interval
    if args.start < overlap_start or args.start + args.duration > overlap_end:
        raise SystemExit(
            f"requested [{args.start}, {args.start + args.duration}] outside overlap "
            f"[{overlap_start}, {overlap_end}]"
        )

    low = [int(value) for value in args.hsv_low.split(",")]
    high = [int(value) for value in args.hsv_high.split(",")]
    captures = [cv2.VideoCapture(str(stream.video_path)) for stream in session.streams]
    tracknet_detectors = None
    if args.detector in ("tracknet", "hybrid"):
        from cv.detectors.tracknet import TrackNetBallDetector
        model_path = Path(args.tracknet_model)
        if not model_path.is_absolute():
            model_path = ROOT / model_path
        tracknet_detectors = [
            TrackNetBallDetector(model_path=str(model_path),
                                 conf_threshold=args.tracknet_threshold)
            for _ in session.streams
        ]
        if any(detector.N_INPUT_FRAMES != 3 for detector in tracknet_detectors):
            raise SystemExit("this demo currently expects a legacy three-frame TrackNet model")
    labels = []
    points = []
    previous = None
    used_frames_b = set()
    sample_count = round(args.duration * args.rate)
    try:
        for sample in range(sample_count):
            session_time = args.start + sample / args.rate
            frame_indices = [
                stream_a.frame_at_session_time(session_time),
                stream_b.frame_at_session_time(session_time),
            ]
            triplet_a = read_triplet(captures[0], frame_indices[0])
            if any(frame is None for frame in triplet_a):
                continue

            def proposals(camera_index, triplet):
                proposed = []
                if args.detector in ("color", "hybrid"):
                    proposed = ball_candidates(
                        *triplet, low, high, 2.0, 500.0, motion_thr=14,
                        min_circularity=0.18, min_fill_ratio=0.18,
                        max_aspect_ratio=3.5, min_radius=1.5, max_radius=20.0,
                        return_metrics=True,
                    )
                    for candidate in proposed:
                        candidate["source"] = "color_motion"
                if tracknet_detectors is not None:
                    proposed.extend(tracknet_candidate(
                        tracknet_detectors[camera_index], triplet
                    ))
                return proposed

            candidates_a = proposals(0, triplet_a)
            best_option = None
            for sync_delta in range(-args.sync_search_frames, args.sync_search_frames + 1):
                candidate_frame_b = frame_indices[1] + sync_delta
                if candidate_frame_b in used_frames_b:
                    continue
                triplet_b = read_triplet(captures[1], candidate_frame_b)
                if any(frame is None for frame in triplet_b):
                    continue
                candidates_b = proposals(1, triplet_b)
                option = choose_pair(
                    candidates_a, candidates_b, stream_a.projection_matrix,
                    stream_b.projection_matrix, args.max_reprojection, previous,
                )
                if option is None:
                    continue
                option_score = option["score"] + 0.15 * abs(sync_delta)
                if best_option is None or option_score < best_option["option_score"]:
                    best_option = {
                        "match": option,
                        "triplet_b": triplet_b,
                        "candidates_b": candidates_b,
                        "frame_b": candidate_frame_b,
                        "sync_delta": sync_delta,
                        "option_score": option_score,
                    }
            if best_option is None:
                continue
            match = best_option["match"]
            triplets = [triplet_a, best_option["triplet_b"]]
            candidates = [candidates_a, best_option["candidates_b"]]
            frame_indices[1] = best_option["frame_b"]
            used_frames_b.add(frame_indices[1])
            previous = match["point"]
            preview_name = f"pair_{sample:04d}_{session_time:.3f}.jpg"
            write_pair_preview(
                previews / preview_name, triplets[0][1], triplets[1][1],
                candidates[0], candidates[1], match,
                (stream_a.camera_id, stream_b.camera_id), session_time,
            )
            frame_image_names = []
            for camera_index, triplet in enumerate(triplets):
                frame_image_name = (
                    f"camera_{camera_index}_{sample:04d}_{session_time:.3f}.jpg"
                )
                cv2.imwrite(str(label_frames / frame_image_name), triplet[1])
                frame_image_names.append(f"frames/{frame_image_name}")
            point = match["point"]
            points.append({
                "session_time": round(session_time, 6),
                "frames": frame_indices,
                "pixels": [list(match["pixel_a"]), list(match["pixel_b"])],
                "world_xyz": [float(value) for value in point],
                "reprojection_px": [float(value) for value in match["errors"]],
                "camera_b_sync_delta_frames": best_option["sync_delta"],
                "preview": f"previews/{preview_name}",
            })
            sequence_id = f"{args.session}:{args.start:.3f}-{args.start + args.duration:.3f}"
            for stream, frame_index, pixel, frame_image in zip(
                    session.streams, frame_indices, (match["pixel_a"], match["pixel_b"]),
                    frame_image_names):
                labels.append({
                    "frame": int(frame_index),
                    "timestamp": round(session_time, 6),
                    "t": round(session_time, 6),
                    "image": frame_image,
                    "state": "visible",
                    "center": [round(pixel[0], 3), round(pixel[1], 3)],
                    "event_tags": [],
                    "sequence_id": sequence_id,
                    "camera_id": stream.camera_id,
                    "metadata": {
                        "review_status": args.review_status,
                        "source_sport": "tennis",
                        "not_padel_ground_truth": True,
                    },
                })
    finally:
        for capture in captures:
            capture.release()

    reprojections = [max(point["reprojection_px"]) for point in points]
    speeds = []
    for previous_point, current_point in zip(points, points[1:]):
        dt = current_point["session_time"] - previous_point["session_time"]
        if dt > 0:
            distance = float(np.linalg.norm(
                np.asarray(current_point["world_xyz"]) - np.asarray(previous_point["world_xyz"])
            ))
            speeds.append(distance / dt)
    summary = {
        "session_id": args.session,
        "window_session_seconds": [args.start, args.start + args.duration],
        "sample_rate_hz": args.rate,
        "samples_attempted": sample_count,
        "matched_samples": len(points),
        "match_rate": len(points) / sample_count if sample_count else 0.0,
        "median_max_reprojection_px": float(np.median(reprojections)) if reprojections else None,
        "p95_max_reprojection_px": float(np.percentile(reprojections, 95)) if reprojections else None,
        "median_speed_mps": float(np.median(speeds)) if speeds else None,
        "max_speed_mps": max(speeds) if speeds else None,
        "calibration_reprojection_px": [
            stream_a.calibration_reprojection_px, stream_b.calibration_reprojection_px,
        ],
        "camera_start_offsets_seconds": [
            stream_a.start_offset_seconds, stream_b.start_offset_seconds,
        ],
        "detector": args.detector,
        "sync_search_frames": args.sync_search_frames,
        "review_status": args.review_status,
        "status": "engineering_proof_not_padel_accuracy",
    }
    (output / "triangulation.json").write_text(json.dumps({
        "summary": summary, "points": points,
    }, indent=2))
    (output / "labels.json").write_text(json.dumps({
        "schema_version": "1.0",
        "coordinate_space": "original_video_pixels",
        "dataset": "CalTennis",
        "review_status": args.review_status,
        "labels": labels,
    }, indent=2))
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"output: {output}")
    return 0 if points else 2


if __name__ == "__main__":
    raise SystemExit(main())
