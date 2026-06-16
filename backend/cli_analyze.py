#!/usr/bin/env python
"""CLI analyze entry — run the full VideoAnalyzer pipeline on a video without the API/frontend.

Mirrors the analysis path in main.py (start_match_analysis), but takes a plain
video file and writes results.json. Useful for testing against datasets such as
PADELVIC.

Examples:
    # Quick smoke test, first 300 frames, auto-detect court
    python cli_analyze.py data/datasets/padelvic/cameras/gopro.mp4 --max-frames 300

    # Full run with a specific detector + annotated output
    python cli_analyze.py video.mp4 --detector tracknet --annotated out.mp4

    # Use an explicit calibration (4 corners or 12 keypoints)
    python cli_analyze.py video.mp4 --calib calib.json
"""
import sys
import os

# Ensure src/ is on Python path for consistent imports (matches main.py).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import argparse
import json
import time

import cv2
import numpy as np

from cv.court_calibration import CourtCalibration
from cv.court_detector import CourtDetector
from models.config import EventDetectorConfig
from models.types import MatchConfig, ServerInfo, TeamId, MatchFormat
from pipeline.video_analyzer import VideoAnalyzer


def _build_match_config(mc_path: str | None, first_server_side: str | None):
    """Construct a MatchConfig from a JSON file and/or a first-server side flag.

    Returns (match_config or None, description).
    """
    if mc_path is None and first_server_side is None:
        return None, "default (server = near side / TEAM_A)"

    data = {}
    if mc_path:
        with open(mc_path) as f:
            data = json.load(f)

    mc = MatchConfig()
    if "match_name" in data:
        mc.match_name = data["match_name"]
    if "golden_point" in data:
        mc.golden_point = bool(data["golden_point"])
    if data.get("format"):
        mc.format = MatchFormat[data["format"]] if data["format"] in MatchFormat.__members__ else mc.format
    if data.get("players"):
        mc.players = data["players"]
    if data.get("teams"):
        mc.teams = {TeamId[k] if k in TeamId.__members__ else TeamId(k): v
                    for k, v in data["teams"].items()}

    # first server: explicit flag wins; "near" => TEAM_A, "far" => TEAM_B
    side = first_server_side or data.get("first_server_side")
    if side:
        team = TeamId.TEAM_A if side == "near" else TeamId.TEAM_B
        players = mc.teams.get(team, ["P1"])
        mc.first_server = ServerInfo(team_id=team, player_id=players[0])
        return mc, f"server = {side} side ({team.value}, {players[0]})"
    if data.get("first_server"):
        fs = data["first_server"]
        team = TeamId[fs["team_id"]] if fs.get("team_id") in TeamId.__members__ else TeamId.TEAM_A
        mc.first_server = ServerInfo(team_id=team, player_id=fs.get("player_id", "P1"))
        return mc, f"server = {team.value}/{mc.first_server.player_id}"
    return mc, "match config (default server)"


def _build_calibration(video_path: str, calib_path: str | None, auto_detect: bool):
    """Pick the best available calibration for a standalone video.

    Priority: explicit --calib JSON > --auto-detect court keypoints > uncalibrated.
    Returns (calibration, description).
    """
    if calib_path:
        with open(calib_path) as f:
            data = json.load(f)
        # 3D CameraModel path — enables ball height (z) estimation
        if data.get("camera_model_keypoints"):
            from cv.camera_model import CameraModel
            cam = CameraModel()
            cam.calibrate(
                data["camera_model_keypoints"],
                net_top_2d=data.get("net_top"),
                image_width=data.get("image_width", 1280),
                image_height=data.get("image_height", 720),
            )
            n = len(data["camera_model_keypoints"])
            has3d = cam.rvec is not None
            return cam, (f"3D CameraModel from {n} keypoints"
                         f"{' +net-top' if data.get('net_top') else ''}"
                         f" (PnP {'ok' if has3d else 'FAILED'})")
        cal = CourtCalibration()
        corners = data.get("corners") or data.get("keypoints")
        if not corners:
            raise ValueError("--calib JSON must contain 'corners' or 'keypoints'")
        if len(corners) == 4:
            net = data.get("net_points")
            cal.calibrate(
                np.array(corners, dtype=np.float32),
                np.array(net, dtype=np.float32) if net else None,
            )
        else:
            cal.calibrate_keypoints(corners)
        return cal, f"explicit calibration from {calib_path}"

    if auto_detect:
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError(f"Cannot read first frame of {video_path}")
        keypoints = CourtDetector().detect(frame)
        if keypoints:
            cal = CourtCalibration()
            cal.calibrate_keypoints(keypoints)
            return cal, f"auto-detected {len(keypoints)} court keypoints"
        print("WARN: court auto-detection failed; running uncalibrated "
              "(positions stay in pixel space)", file=sys.stderr)

    return CourtCalibration(), "uncalibrated (pixel-space positions)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run padel analysis on a video file.")
    parser.add_argument("video", help="Path to input video")
    parser.add_argument("--detector", choices=["yolo", "tracknet", "fast"],
                        default="yolo", help="Ball detector (default: yolo)")
    parser.add_argument("--out", help="Output results JSON path "
                        "(default: <video>.results.json)")
    parser.add_argument("--annotated", help="Optional annotated video output path")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (quick test on large videos)")
    parser.add_argument("--calib", help="Calibration JSON "
                        "('corners' [4] or 'keypoints' [>=4], optional 'net_points')")
    parser.add_argument("--no-auto-detect", action="store_true",
                        help="Skip court auto-detection when no --calib given")
    parser.add_argument("--ball-conf", type=float, default=None,
                        help="Override ball detector confidence threshold "
                        "(tracknet/fast). Lower = more detections (e.g. 0.35)")
    parser.add_argument("--match-config", help="Match config JSON "
                        "(teams, first_server, golden_point, format)")
    parser.add_argument("--first-server", choices=["near", "far"], default=None,
                        help="Which baseline serves first (near=TEAM_A, far=TEAM_B)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        return 1

    out_path = args.out or (os.path.splitext(args.video)[0] + ".results.json")

    cal, cal_desc = _build_calibration(
        args.video, args.calib, auto_detect=not args.no_auto_detect
    )
    match_config, mc_desc = _build_match_config(args.match_config, args.first_server)
    print(f"calibration: {cal_desc}")
    print(f"detector:    {args.detector}")
    print(f"match cfg:   {mc_desc}")
    print(f"video:       {args.video}")

    analyzer = VideoAnalyzer(
        match_id=os.path.basename(args.video),
        calibration=cal,
        config=EventDetectorConfig(),
        match_config=match_config,
        detector_type=args.detector,
    )

    if args.ball_conf is not None and hasattr(analyzer.ball_detector, "_conf_threshold"):
        analyzer.ball_detector._conf_threshold = args.ball_conf
        print(f"ball conf:   {args.ball_conf} (overridden)")

    last = {"pct": -1.0}

    def progress_cb(frame, total, pct):
        if pct - last["pct"] >= 1.0:
            last["pct"] = pct
            print(f"\r  {frame}/{total or '?'} frames ({pct:.0f}%)",
                  end="", file=sys.stderr, flush=True)

    t0 = time.time()
    result = analyzer.analyze_video(
        args.video,
        progress_callback=progress_cb,
        annotated_path=args.annotated,
        max_frames=args.max_frames,
    )
    elapsed = time.time() - t0
    print(file=sys.stderr)

    wall_hits = [
        {
            "event_type": e.event_type.value,
            "frame_number": e.frame_number,
            "timestamp": e.timestamp,
            "metadata": e.metadata,
        }
        for e in analyzer.all_events
        if e.event_type.value == "WALL_HIT"
    ]

    results = {
        "score": analyzer.scoring_engine.get_score_display(),
        "events": [
            {"event_type": e.event_type.value, "timestamp": e.timestamp,
             "frame_number": e.frame_number,
             "position": {"x": e.position.x, "y": e.position.y},
             "metadata": e.metadata}
            for e in analyzer.all_events
        ],
        "wall_hits": wall_hits,
        "trajectory": analyzer.ball_tracker.trajectory,
        "player_positions": analyzer.player_positions_log,
        "frames_processed": result.get("frames_processed", 0),
    }
    with open(out_path, "w") as f:
        json.dump(results, f)

    frames = result.get("frames_processed", 0)
    fps = frames / elapsed if elapsed > 0 else 0
    print(f"\ndone in {elapsed:.1f}s ({fps:.1f} fps)")
    print(f"  frames:     {frames}")
    print(f"  events:     {len(analyzer.all_events)} ({len(wall_hits)} wall hits)")
    print(f"  trajectory: {len(analyzer.ball_tracker.trajectory)} ball points")
    print(f"  results ->  {out_path}")
    if args.annotated:
        print(f"  annotated -> {args.annotated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
