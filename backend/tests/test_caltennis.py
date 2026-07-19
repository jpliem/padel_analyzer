import json

import numpy as np
import pytest

from cv.caltennis import capture_start_seconds, load_session


def _write_calibration(path, camera_x):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "K": [[1000, 0, 960], [0, 1000, 540], [0, 0, 1]],
        "R_w2c": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "t_w2c": [-camera_x, 0, 10],
        "dist_coeffs": None,
        "reprojection_error": 1.25,
    }))


def test_capture_start_seconds_preserves_millisecond_offsets():
    first = capture_start_seconds("01_23_2026_16_59_47_000_camera")
    second = capture_start_seconds("01_23_2026_17_00_17_250_camera")
    assert second - first == pytest.approx(30.25)


def test_load_session_aligns_frames_on_shared_capture_time(tmp_path):
    session_id = "01_23_2026_17_00_court2"
    ids = [
        "01_23_2026_16_59_47_000_2_E",
        "01_23_2026_17_00_17_000_2_W",
    ]
    rows = []
    for index, video_id in enumerate(ids):
        video = tmp_path / session_id / f"{video_id}.mp4"
        timestamps = tmp_path / session_id / f"{video_id}_timestamps.npy"
        calibration = tmp_path / "camera_calibration" / session_id / video_id / "calib.json"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
        np.save(timestamps, np.arange(0, 10_001, 1000, dtype=np.float64))
        _write_calibration(calibration, float(index))
        rows.append({
            "video": str(video.relative_to(tmp_path)),
            "timestamps": str(timestamps.relative_to(tmp_path)),
            "calibration": str(calibration.relative_to(tmp_path)),
            "session_id": session_id,
            "video_id": video_id,
        })
    (tmp_path / "metadata_mini.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    session = load_session(tmp_path, session_id)

    assert [stream.start_offset_seconds for stream in session.streams] == [0.0, 30.0]
    assert session.streams[0].frame_at_session_time(32.0) == 10
    assert session.streams[1].frame_at_session_time(32.0) == 2
    assert session.streams[1].session_time_for_frame(2) == pytest.approx(32.0)
    assert session.streams[0].projection_matrix.shape == (3, 4)
