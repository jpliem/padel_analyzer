#!/usr/bin/env python
"""Feed triangulated 3D ball positions into the existing scoring pipeline.

The scoring brain (EventDetector -> BounceDetector/ServeDetector/PointEnd ->
PadelScoringEngine) already exists; it was only ever starved by single-camera
z=0. This bypasses the single-camera ball tracker and feeds it the clean
two-camera triangulated 3D instead, then writes a results.json that
scripts/eval_rallies.py can grade.

Example:
    python scripts/score_from_3d.py --points /tmp/tri_gated.json \
        --first-server near --out /tmp/score_3d.json
"""
import sys, os, argparse, json, math

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "backend", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", default="/tmp/tri_gated.json")
    ap.add_argument("--first-server", choices=["near", "far"], default="near")
    ap.add_argument("--fps", type=float, default=50.0, help="reference fps for frame numbers")
    ap.add_argument("--out", default="/tmp/score_3d.json")
    args = ap.parse_args()

    os.chdir(os.path.join(_ROOT, "backend"))
    from cv.court_calibration import CourtCalibration
    from cv.player_tracker import PlayerTracker
    from logic.event_detector import EventDetector
    from logic.scoring_engine import PadelScoringEngine
    from models.config import EventDetectorConfig
    from models.court_model import PadelCourtModel
    from models.types import ServerInfo, TeamId

    pts = sorted(json.load(open(args.points))["points"], key=lambda p: p["t"])
    pts = [p for p in pts if p["on_court"]]  # only trust gated, on-court 3D

    cal = CourtCalibration()  # only used for is_in_service_box (pure court geometry)
    court_model = PadelCourtModel()
    team = TeamId.TEAM_A if args.first_server == "near" else TeamId.TEAM_B
    first_server = ServerInfo(team_id=team, player_id="P1" if team == TeamId.TEAM_A else "P3")
    scoring = PadelScoringEngine(golden_point=True, sets_to_win=2,
                                 first_server=first_server,
                                 team_players={TeamId.TEAM_A: ["P1", "P2"],
                                               TeamId.TEAM_B: ["P3", "P4"]})
    team_map = {"P1": 1, "P2": 1, "P3": 2, "P4": 2}
    ed = EventDetector(EventDetectorConfig(), cal, scoring,
                       PlayerTracker(cal), team_map, court_model=court_model)

    events = []
    prev = None
    for p in pts:
        if prev is not None:
            dt = p["t"] - prev["t"]
            dist = math.dist((p["x"], p["y"]), (prev["x"], prev["y"]))
            speed = (dist / dt * 3.6) if dt > 0 else 0.0  # km/h, matches BallTracker
        else:
            speed = 0.0
        fno = int(round(p["t"] * args.fps))
        ball_pos = {"x": p["x"], "y": p["y"], "z": p["z"], "speed": speed,
                    "timestamp": p["t"], "frame": fno, "detected": True}
        for e in ed.process(ball_pos, [], fno):
            events.append({"event_type": e.event_type.value, "frame_number": e.frame_number,
                           "timestamp": e.timestamp,
                           "metadata": getattr(e, "metadata", {}) or {}})
        prev = p

    tally = {}
    for e in events:
        tally[e["event_type"]] = tally.get(e["event_type"], 0) + 1

    last_frame = int(round(pts[-1]["t"] * args.fps)) if pts else 0
    json.dump({"events": events, "frames_processed": last_frame,
               "score": scoring.get_score_display()}, open(args.out, "w"))

    print("=== scoring from triangulated 3D ===")
    print(f"  fed {len(pts)} gated on-court 3D points")
    print(f"  event tally: {tally or '(none)'}")
    print(f"  final score: {scoring.get_score_display()}")
    print(f"  -> {args.out}")
    print(f"  grade with: scripts/eval_rallies.py --results {args.out} "
          f"--xlsx data/datasets/padelvic/derived/PadelVic_Panasonic_labeling.xlsx "
          f"--max-frame {last_frame}")


if __name__ == "__main__":
    main()
