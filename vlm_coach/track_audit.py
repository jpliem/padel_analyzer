"""VLM audit of ball-tracker output: does the marker sit on the actual ball?

Renders annotated frames at decision moments (monocular-fit windows and
detected events), asks a local vision model to judge each one, and writes a
JSON report with per-frame verdicts and a disagreement summary. Disagreement
frames are exactly the frames worth hand-labeling next.

This is an audit layer, not a scorer: verdicts flag frames for review and
labeling; they never change the score.

Usage:
    python -m vlm_coach.track_audit \
        --results /tmp/e2e_panasonic3d.results.json \
        --video data/datasets/padelvic/cameras/panasonic_final.mp4 \
        --calib /tmp/panasonic_calib3d.json \
        --model qwen2.5vl:3b --sample 12 --out /tmp/track_audit.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .ollama_client import OllamaClient
from .schemas import BallPoint, TrackAuditVerdict, model_dump

AUDIT_PROMPT = """You are auditing a padel ball tracker. The image is one frame of an
indoor padel match filmed from a fixed, elevated CCTV-style camera behind the
court. The court has a blue floor, glass walls, black metal fencing, and a net
across the middle. Four players in sportswear are on court. The ball is a small
yellow padel ball, only a few pixels wide at this camera distance, and often
smeared into a faint streak by motion blur.

A RED CIRCLE marks where the tracker believes the ball is. Faded gray dots show
its recent trail. Look carefully INSIDE the red circle first: a tiny bright or
yellowish dot or short streak there is likely the ball. Judge only what you can
see:

- marker_on_ball: "yes" if the red circle is on the ball, "close" if within
  about one ball-width, "no" if it marks something else (player, racket, shoe,
  glass reflection, empty court), "no_ball_visible" if you cannot find any ball
  in the frame, "unclear" if you cannot tell.
- ball_visible_elsewhere: true only if you can clearly see the ball somewhere
  the marker is not.
- ball_location_hint: short phrase locating the real ball if visible
  (e.g. "above net, left side").

Balls are small, yellow, and often motion-blurred. Rackets and shoes are not
balls. Be strict: when in doubt, use "unclear" rather than guessing."""

POINT_PROMPT = """This is a crop from an indoor padel match video (blue court,
glass walls, players in sportswear). There may be a small yellow padel ball in
the image, possibly motion-blurred into a faint streak. If you can see it,
return found=true and its position in 0-1000 normalized coordinates (x from
left edge, y from top edge). If no ball is visible, return found=false. Rackets,
shoes, lights, and reflections are not the ball."""

# Verdict thresholds in normalized (0-1000) crop coordinates: at crop 480 on a
# 4K source, one ball-width is roughly 30 units.
POINT_AGREE = 60
POINT_CLOSE = 130


def point_to_verdict(point: BallPoint, marker_norm: tuple) -> TrackAuditVerdict:
    """Map a pointed ball location to the audit verdict vocabulary. Distances
    are computed in code — the VLM only locates, it never judges the tracker."""
    if not point.found:
        return TrackAuditVerdict(marker_on_ball="no_ball_visible",
                                 confidence=point.confidence)
    distance = float(np.hypot(point.x - marker_norm[0], point.y - marker_norm[1]))
    if distance <= POINT_AGREE:
        state = "yes"
    elif distance <= POINT_CLOSE:
        state = "close"
    else:
        state = "no"
    return TrackAuditVerdict(
        marker_on_ball=state,
        ball_visible_elsewhere=state == "no",
        ball_location_hint=f"pointed at ({point.x}, {point.y})/1000, "
                           f"{distance:.0f} units from marker",
        confidence=point.confidence,
    )


def load_ground_homography(calib_path: str) -> np.ndarray:
    calib = json.loads(Path(calib_path).read_text())
    src_m = np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=np.float64)
    dst_px = np.array(calib["corners"], dtype=np.float64)
    homography, _ = cv2.findHomography(src_m, dst_px)
    return homography


def court_to_pixel(homography: np.ndarray, x: float, y: float) -> Optional[tuple]:
    point = cv2.perspectiveTransform(
        np.array([[[x, y]]], dtype=np.float64), homography)[0][0]
    if not np.all(np.isfinite(point)):
        return None
    return float(point[0]), float(point[1])


def pick_audit_frames(results: Dict, sample: int) -> List[int]:
    """Prefer frames where the pipeline made a claim worth checking."""
    trajectory = {p["frame"]: p for p in results.get("trajectory", [])}
    fit_frames = [f for f, p in trajectory.items()
                  if p.get("position_source") == "monocular_ballistic_fit"]
    event_frames = [int(e.get("frame_number", e.get("frame", 0)))
                    for e in results.get("events", [])]
    detected = [f for f, p in trajectory.items() if p.get("detected")]

    ordered: List[int] = []
    for pool in (event_frames, fit_frames, detected):
        step = max(1, len(pool) // max(1, sample // 3))
        ordered.extend(sorted(set(pool))[::step])
    unique = sorted(set(f for f in ordered if f in trajectory))
    step = max(1, len(unique) // sample) if len(unique) > sample else 1
    return unique[::step][:sample]


def crop_around(frame: np.ndarray, center_px: tuple, crop: int) -> np.ndarray:
    """Crop a square window around the marker so the tiny ball is visible to
    the VLM. Full-frame audits score near-zero: the ball is only a few pixels
    wide at court-camera distance."""
    height, width = frame.shape[:2]
    half = crop // 2
    left = int(min(max(center_px[0] - half, 0), max(width - crop, 0)))
    top = int(min(max(center_px[1] - half, 0), max(height - crop, 0)))
    return frame[top:top + crop, left:left + crop]


def render_audit_frame(frame: np.ndarray, trajectory: Dict[int, Dict],
                       homography: np.ndarray, frame_no: int,
                       crop: int = 0) -> Optional[np.ndarray]:
    point = trajectory.get(frame_no)
    if point is None:
        return None
    for past in range(max(0, frame_no - 30), frame_no):
        trail = trajectory.get(past)
        if trail is None:
            continue
        pixel = court_to_pixel(homography, trail["x"], trail["y"])
        if pixel:
            cv2.circle(frame, (int(pixel[0]), int(pixel[1])), 5, (180, 180, 180), -1)
    pixel = court_to_pixel(homography, point["x"], point["y"])
    if pixel is None:
        return None
    cv2.circle(frame, (int(pixel[0]), int(pixel[1])), 16, (0, 0, 255), 4)
    if crop > 0:
        return crop_around(frame, pixel, crop)
    height, width = frame.shape[:2]
    if width > 1600:
        scale = 1600.0 / width
        frame = cv2.resize(frame, (1600, int(height * scale)))
    return frame


def audit(results_path: str, video_path: str, calib_path: str, model: str,
          sample: int, out_dir: Path, client: Optional[OllamaClient] = None,
          crop: int = 480, mode: str = "pointing") -> Dict:
    """Audit tracker claims with a local VLM.

    mode="pointing" (default): crop around the marker, ask the model to point
    at the ball, derive the verdict from the pointed-vs-marker distance in
    code. Small local models point far more reliably than they judge.
    mode="judgment": legacy behaviour — render the marker and ask the model
    to rate it directly.
    """
    results = json.loads(Path(results_path).read_text())
    trajectory = {p["frame"]: p for p in results.get("trajectory", [])}
    homography = load_ground_homography(calib_path)
    frames = pick_audit_frames(results, sample)
    client = client or OllamaClient()

    out_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(video_path)
    verdicts: List[Dict] = []
    for frame_no in frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = capture.read()
        if not ok:
            continue
        if mode == "pointing" and crop > 0:
            point = trajectory.get(frame_no)
            pixel = point and court_to_pixel(homography, point["x"], point["y"])
            if not pixel:
                continue
            rendered = crop_around(frame, pixel, crop)
            height, width = frame.shape[:2]
            half = crop // 2
            left = min(max(pixel[0] - half, 0), max(width - crop, 0))
            top = min(max(pixel[1] - half, 0), max(height - crop, 0))
            marker_norm = ((pixel[0] - left) / crop * 1000.0,
                           (pixel[1] - top) / crop * 1000.0)
            image_path = out_dir / f"audit_{frame_no:06d}.jpg"
            cv2.imwrite(str(image_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 92])
            ball_point = client.structured(model, POINT_PROMPT, BallPoint,
                                           images=[image_path])
            verdict = point_to_verdict(ball_point, marker_norm)
        else:
            rendered = render_audit_frame(frame, trajectory, homography, frame_no,
                                          crop=crop)
            if rendered is None:
                continue
            image_path = out_dir / f"audit_{frame_no:06d}.jpg"
            cv2.imwrite(str(image_path), rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
            verdict = client.structured(model, AUDIT_PROMPT, TrackAuditVerdict,
                                        images=[image_path])
        claim = trajectory[frame_no]
        verdicts.append({
            "frame": frame_no,
            "claim": {
                "x": claim["x"], "y": claim["y"], "z": claim.get("z"),
                "source": claim.get("position_source", "kalman"),
                "detected": claim.get("detected"),
            },
            "verdict": model_dump(verdict),
            "image": str(image_path),
        })

    capture.release()
    agreed = [v for v in verdicts if v["verdict"]["marker_on_ball"] in ("yes", "close")]
    disagreed = [v for v in verdicts if v["verdict"]["marker_on_ball"] == "no"]
    report = {
        "model": model,
        "mode": mode,
        "frames_audited": len(verdicts),
        "agreed": len(agreed),
        "disagreed": len(disagreed),
        "no_ball_visible": sum(
            1 for v in verdicts if v["verdict"]["marker_on_ball"] == "no_ball_visible"),
        "unclear": sum(
            1 for v in verdicts if v["verdict"]["marker_on_ball"] == "unclear"),
        "label_queue": [v["frame"] for v in disagreed],
        "note": "Audit verdicts flag frames for review/labeling; they never change the score.",
        "verdicts": verdicts,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="VLM audit of tracker output")
    parser.add_argument("--results", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--calib", required=True)
    parser.add_argument("--model", default="qwen3-vl:2b-instruct")
    parser.add_argument("--sample", type=int, default=12)
    parser.add_argument("--crop", type=int, default=480,
                        help="Crop size (px, source resolution) around the "
                             "marker; 0 audits the full frame")
    parser.add_argument("--mode", default="pointing",
                        choices=["pointing", "judgment"],
                        help="pointing: VLM locates the ball, verdict derived "
                             "in code (recommended); judgment: legacy direct "
                             "marker rating")
    parser.add_argument("--out", default="/tmp/track_audit.json")
    parser.add_argument("--frames-dir", default="/tmp/track_audit_frames")
    args = parser.parse_args()

    report = audit(args.results, args.video, args.calib, args.model,
                   args.sample, Path(args.frames_dir), crop=args.crop,
                   mode=args.mode)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"audited {report['frames_audited']} frames: "
          f"{report['agreed']} agreed, {report['disagreed']} disagreed, "
          f"{report['no_ball_visible']} no-ball, {report['unclear']} unclear")
    print(f"label queue: {report['label_queue']}")
    print(f"report -> {args.out}")


if __name__ == "__main__":
    main()
