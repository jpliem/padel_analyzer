import json
from pathlib import Path

import cv2
import numpy as np

from vlm_coach.schemas import TrackAuditVerdict
from vlm_coach.track_audit import (
    audit, court_to_pixel, load_ground_homography, pick_audit_frames,
    render_audit_frame,
)


def _calib_file(tmp_path: Path) -> Path:
    path = tmp_path / "calib.json"
    path.write_text(json.dumps({
        "corners": [[100, 900], [1180, 900], [980, 200], [300, 200]],
    }))
    return path


def _results(frames):
    return {
        "trajectory": [
            {"frame": f, "x": 5.0, "y": 10.0, "z": 1.0, "detected": True,
             "position_source": "monocular_ballistic_fit" if f % 2 == 0 else "kalman"}
            for f in frames
        ],
        "events": [{"event_type": "BOUNCE", "frame_number": frames[0]}],
    }


def test_ground_homography_round_trip(tmp_path):
    homography = load_ground_homography(str(_calib_file(tmp_path)))
    corner = court_to_pixel(homography, 0, 0)
    assert corner is not None
    assert abs(corner[0] - 100) < 1e-6 and abs(corner[1] - 900) < 1e-6


def test_pick_audit_frames_prefers_claims_and_caps_sample():
    picked = pick_audit_frames(_results(list(range(0, 200, 2))), sample=10)
    assert 0 < len(picked) <= 10
    assert all(isinstance(f, int) for f in picked)


def test_render_audit_frame_draws_marker(tmp_path):
    homography = load_ground_homography(str(_calib_file(tmp_path)))
    trajectory = {10: {"frame": 10, "x": 5.0, "y": 10.0, "z": 1.0}}
    frame = np.zeros((1080, 1280, 3), dtype=np.uint8)
    rendered = render_audit_frame(frame, trajectory, homography, 10)
    assert rendered is not None
    assert rendered[..., 2].max() == 255  # red marker present


def test_audit_uses_client_and_summarizes(tmp_path, monkeypatch):
    frames = list(range(0, 30, 3))
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_results(frames)))
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25, (1280, 1080))
    for _ in range(40):
        writer.write(np.zeros((1080, 1280, 3), dtype=np.uint8))
    writer.release()

    class FakeClient:
        def __init__(self):
            self.calls = 0

        def structured(self, model, prompt, output_type, images=()):
            self.calls += 1
            state = "no" if self.calls == 1 else "yes"
            return TrackAuditVerdict(marker_on_ball=state, confidence=0.9)

    fake = FakeClient()
    report = audit(str(results_path), str(video_path), str(_calib_file(tmp_path)),
                   model="fake", sample=4, out_dir=tmp_path / "frames", client=fake,
                   mode="judgment")
    assert report["frames_audited"] == fake.calls > 0
    assert report["disagreed"] == 1
    assert report["label_queue"]  # first frame flagged for labeling
    assert report["agreed"] == report["frames_audited"] - 1


def test_audit_pointing_mode_maps_distance_to_verdict(tmp_path):
    frames = list(range(0, 30, 3))
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(_results(frames)))
    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             25, (1280, 1080))
    for _ in range(40):
        writer.write(np.zeros((1080, 1280, 3), dtype=np.uint8))
    writer.release()

    from vlm_coach.schemas import BallPoint

    class FakePointer:
        def __init__(self):
            self.calls = 0

        def structured(self, model, prompt, output_type, images=()):
            assert output_type is BallPoint
            self.calls += 1
            if self.calls == 1:
                return BallPoint(found=False)          # -> no_ball_visible
            if self.calls == 2:
                return BallPoint(found=True, x=500, y=500)  # centre -> yes
            return BallPoint(found=True, x=950, y=950)      # far -> no

    fake = FakePointer()
    report = audit(str(results_path), str(video_path), str(_calib_file(tmp_path)),
                   model="fake", sample=12, out_dir=tmp_path / "frames",
                   client=fake, crop=320, mode="pointing")
    verdicts = [v["verdict"]["marker_on_ball"] for v in report["verdicts"]]
    assert len(verdicts) >= 3
    assert verdicts[0] == "no_ball_visible"
    assert verdicts[1] == "yes"
    assert all(v == "no" for v in verdicts[2:])
    assert report["no_ball_visible"] == 1
    assert report["disagreed"] == len(verdicts) - 2
    flagged = report["label_queue"]
    assert report["verdicts"][2]["frame"] in flagged
