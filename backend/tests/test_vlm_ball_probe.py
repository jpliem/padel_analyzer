import json
import subprocess
import sys

from scripts.vlm_ball_probe import extract_video_points, summarize_against_gt


def test_extract_video_points_scales_molmo_coordinates():
    text = '<points coords="0: 1 500 250; 1: 1 750 500"/>'

    points = extract_video_points(text, image_w=1280, image_h=720)

    assert points == [
        {"frame": 0.0, "id": "1", "x": 640.0, "y": 180.0},
        {"frame": 1.0, "id": "1", "x": 960.0, "y": 360.0},
    ]


def test_summarize_against_gt_reports_pixel_errors():
    points = [
        {"frame": 0.0, "id": "1", "x": 100.0, "y": 200.0},
        {"frame": 1.0, "id": "1", "x": 130.0, "y": 220.0},
    ]
    gt = [(103.0, 204.0), (190.0, 220.0)]

    summary = summarize_against_gt(points, gt, threshold_px=15)

    assert summary["matched_points"] == 2
    assert summary["mean_error_px"] == 32.5
    assert summary["median_error_px"] == 32.5
    assert summary["pck"] == 0.5


def test_cli_can_parse_raw_text_without_video(tmp_path):
    raw = tmp_path / "molmo.txt"
    out = tmp_path / "points.json"
    raw.write_text('<points coords="0: 1 500 250"/>')

    result = subprocess.run(
        [
            sys.executable,
            "scripts/vlm_ball_probe.py",
            "--raw-text",
            str(raw),
            "--image-width",
            "1280",
            "--image-height",
            "720",
            "--out",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    data = json.loads(out.read_text())
    assert data["points"] == [{"frame": 0.0, "id": "1", "x": 640.0, "y": 180.0}]
