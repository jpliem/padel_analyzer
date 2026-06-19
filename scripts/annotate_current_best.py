#!/usr/bin/env python
"""Annotate the current best diagnostic view: 3D ball + players + score/events."""
import argparse
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))

import cv2


def point_by_frame(points, fps):
    return {
        int(round(p["t"] * fps)): p
        for p in points
        if p.get("on_court", True)
    }


def nearest_point(indexed_points, frame_number, max_delta=3):
    for delta in (0, -1, 1, -2, 2, -3, 3):
        if abs(delta) > max_delta:
            continue
        point = indexed_points.get(frame_number + delta)
        if point:
            return point
    return None


def score_overlay_text(score, mode):
    return f"{score.get('score')} | G {score.get('games')} | S {score.get('sets')} | {mode}"


def event_tally(events):
    tally = {}
    for event in events:
        kind = event.get("event_type", "UNKNOWN")
        tally[kind] = tally.get(kind, 0) + 1
    return tally


def resolve_repo_path(path):
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


def latest_event_label(events, frame_number):
    previous = [e for e in events if int(e.get("frame_number", -1)) <= frame_number]
    if not previous:
        return ""
    event = max(previous, key=lambda e: int(e.get("frame_number", -1)))
    return f"{event.get('event_type', 'UNKNOWN')} @ f{int(event.get('frame_number', 0))}"


def draw_text_box(frame, text, origin, scale=0.75, color=(255, 255, 255)):
    x, y = origin
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(frame, (x - 8, y - th - 10), (x + tw + 8, y + 8), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def make_player_detector(model_path):
    from cv.detectors.yolo import UnifiedYoloDetector, YoloPlayerDetector
    return YoloPlayerDetector(UnifiedYoloDetector(model_path), conf_threshold=0.4)


def load_camera(calib_path):
    from cv.camera_model import CameraModel
    data = json.load(open(calib_path))
    cam = CameraModel()
    cam.calibrate(
        data["camera_model_keypoints"],
        image_width=data.get("image_width", 1920),
        image_height=data.get("image_height", 1080),
    )
    return cam


def draw_players(frame, detections):
    for det in detections:
        x1, y1, x2, y2, conf = det[:5]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 185, 116), 2)
        cv2.putText(
            frame,
            f"player {conf:.2f}",
            (int(x1), max(20, int(y1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 185, 116),
            2,
        )


def draw_ball(frame, cam, point, trail):
    if point is not None:
        u, v = cam.court_to_pixel(point["x"], point["y"], point["z"])
        trail.append((u, v, point["z"]))
        if len(trail) > 18:
            trail.pop(0)

    for i, (u, v, z) in enumerate(trail):
        alpha = (i + 1) / max(len(trail), 1)
        cv2.circle(frame, (int(u), int(v)), max(3, int(5 + 7 * alpha)),
                   (0, int(210 * alpha), 255), -1)

    if trail:
        u, v, z = trail[-1]
        cv2.circle(frame, (int(u), int(v)), 16, (0, 0, 255), 3)
        cv2.putText(frame, f"3D ball z={z:.2f}m", (int(u) + 18, int(v) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"could not open video writer for {path}")


def writer_fps(source_fps):
    return source_fps if source_fps and source_fps > 0 else 30.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/datasets/padelvic/cameras/panasonic_final.mp4")
    parser.add_argument("--calib", default="/tmp/panasonic_cammodel2.json")
    parser.add_argument("--points", default="/tmp/tri_matched_current_best.json")
    parser.add_argument("--score", default="/tmp/score_3d_current_best.json")
    parser.add_argument("--start", type=float, default=2.0)
    parser.add_argument("--window", type=float, default=20.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--player-every", type=int, default=3,
                        help="Run YOLO player detection every N output frames.")
    parser.add_argument("--yolo-model", default="backend/yolov8n.pt")
    parser.add_argument("--out", default="/tmp/current_best_overlay.mp4")
    args = parser.parse_args()

    video = resolve_repo_path(args.video)
    points_data = json.load(open(args.points))
    score_data = json.load(open(args.score)) if os.path.exists(args.score) else {
        "score": {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"},
        "events": [],
        "mode": "unknown",
    }

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = args.width / in_w
    out_size = (args.width, int(in_h * scale))
    writer = open_writer(args.out, writer_fps(fps), out_size)

    cam = load_camera(args.calib)
    player_detector = make_player_detector(resolve_repo_path(args.yolo_model))
    indexed_points = point_by_frame(points_data.get("points", []), fps)
    events = score_data.get("events", [])
    tally = event_tally(events)
    score_text = score_overlay_text(score_data.get("score", {}), score_data.get("mode", "unknown"))

    f0 = int(args.start * fps)
    f1 = int((args.start + args.window) * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    trail = []
    last_players = []
    frames_written = 0
    ball_frames = 0
    player_frames = 0

    for fno in range(f0, f1):
        ok, frame = cap.read()
        if not ok:
            break
        if frames_written % max(args.player_every, 1) == 0:
            detections = player_detector.detect(frame, fno)
            last_players = detections.tolist() if len(detections) else []
        if last_players:
            player_frames += 1
        draw_players(frame, last_players)

        point = nearest_point(indexed_points, fno)
        if point is not None:
            ball_frames += 1
        draw_ball(frame, cam, point, trail)

        draw_text_box(frame, score_text, (30, 42), scale=0.8)
        event_text = "events " + ", ".join(f"{k}:{v}" for k, v in sorted(tally.items()))
        draw_text_box(frame, event_text, (30, 82), scale=0.65, color=(0, 255, 255))
        latest = latest_event_label(events, fno)
        if latest:
            draw_text_box(frame, f"latest {latest}", (30, 118), scale=0.65, color=(0, 200, 255))
        cv2.putText(frame, f"t={fno / fps:.2f}s  current-best diagnostic overlay",
                    (30, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        writer.write(cv2.resize(frame, out_size))
        frames_written += 1

    cap.release()
    writer.release()
    print(f"wrote {args.out}")
    print(f"  frames: {frames_written}")
    print(f"  ball frames drawn: {ball_frames}")
    print(f"  player-overlay frames: {player_frames}")
    print(f"  events: {tally or '(none)'}")


if __name__ == "__main__":
    main()
