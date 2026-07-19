import io
import json
import os
import shutil

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def smart_client(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "matches"))
    main._active_analyzers.clear()
    main._analysis_jobs.clear()
    return TestClient(main.app), main, tmp_path


def _create(client):
    response = client.post("/match/setup", json={
        "match_name": "Single camera demo",
        "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
        "teams": {"TEAM_A": ["P1", "P2"], "TEAM_B": ["P3", "P4"]},
        "golden_point": True, "format": "best_of_3",
    })
    return response.json()["match_id"]


def _video_bytes(tmp_path):
    path = str(tmp_path / "tiny.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for index in range(10):
        writer.write(np.full((48, 64, 3), index * 10, dtype=np.uint8))
    writer.release()
    with open(path, "rb") as handle:
        return handle.read()


def _saved_results(main, match_id):
    results = {
        "score": {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"},
        "events": [
            {"event_type": "SERVE", "timestamp": 1.0, "frame_number": 10,
             "position": {"x": 5, "y": 3}, "confidence": .9, "metadata": {}},
            {"event_type": "POINT_END", "timestamp": 8.0, "frame_number": 80,
             "position": {"x": 5, "y": 17}, "confidence": .7,
             "metadata": {"reason": "out"}},
        ],
        "trajectory": [], "player_positions": [], "reviews": [], "frames_processed": 100,
        "media": {"fps": 10, "duration_seconds": 10, "original_name": "tiny.avi"},
    }
    results["highlights"] = main._build_highlights(results["events"], results["media"])
    results["stats"] = main._build_stats(results)
    main._save_results(match_id, results)


def test_upload_persists_video_metadata_and_status(smart_client):
    client, main, tmp_path = smart_client
    match_id = _create(client)
    response = client.post(
        f"/analyze/upload?match_id={match_id}",
        files={"file": ("tiny.avi", io.BytesIO(_video_bytes(tmp_path)), "video/x-msvideo")},
    )
    assert response.status_code == 200
    assert response.json()["media"]["fps"] == 10.0
    assert response.json()["media"]["duration_seconds"] == 1.0
    main._analysis_jobs.clear()
    status = client.get(f"/analyze/status/{match_id}").json()
    assert status["state"] == "uploaded"


def test_result_highlights_exports_and_recording(smart_client):
    client, main, tmp_path = smart_client
    match_id = _create(client)
    client.post(
        f"/analyze/upload?match_id={match_id}",
        files={"file": ("tiny.avi", io.BytesIO(_video_bytes(tmp_path)), "video/x-msvideo")},
    )
    _saved_results(main, match_id)
    main._save_job(match_id, {"state": "complete", "percent": 100})

    result = client.get(f"/match/{match_id}/result").json()
    assert result["analysis"]["stats"]["rallies"] == 1
    assert result["analysis"]["highlights"][0]["needs_review"] is True
    assert client.get(f"/match/{match_id}/recording").status_code == 200
    assert client.get(f"/match/{match_id}/export.json").status_code == 200
    csv_response = client.get(f"/match/{match_id}/export.csv")
    assert csv_response.status_code == 200
    assert "POINT_END" in csv_response.text
    if shutil.which("ffmpeg"):
        clip = client.get(f"/match/{match_id}/highlights/rally-1.mp4")
        assert clip.status_code == 200
        assert clip.headers["content-type"] == "video/mp4"
        assert len(clip.content) > 100


def test_saved_score_correction_survives_without_analyzer(smart_client):
    client, main, _ = smart_client
    match_id = _create(client)
    _saved_results(main, match_id)
    main._active_analyzers.clear()
    response = client.post(f"/match/{match_id}/correct-score", json={"team": 1})
    assert response.status_code == 200
    assert response.json()["score"]["score"] == "15 - 0"
    main._active_analyzers.clear()
    assert client.get(f"/match/{match_id}/score").json()["score"] == "15 - 0"


def test_interrupted_processing_job_becomes_actionable_error(smart_client):
    client, main, _ = smart_client
    match_id = _create(client)
    main._save_job(match_id, {"state": "processing", "percent": 42})
    main._analysis_jobs.clear()
    status = client.get(f"/analyze/status/{match_id}").json()
    assert status["state"] == "error"
    assert "restart" in status["error"]


def test_legacy_result_is_upgraded_for_webapp(smart_client):
    client, main, _ = smart_client
    match_id = _create(client)
    main._write_json(os.path.join(main._match_dir(match_id), "results.json"), {
        "score": {"score": "15 - 0", "games": "0 - 0", "sets": "0 - 0"},
        "events": [], "trajectory": [], "player_positions": [], "frames_processed": 12,
    })
    response = client.get(f"/match/{match_id}/result")
    assert response.status_code == 200
    analysis = response.json()["analysis"]
    assert analysis["reviews"] == []
    assert analysis["highlights"] == []
    assert analysis["stats"]["frames_processed"] == 12
    assert analysis["media"]["fps"] == 30.0
    assert analysis["model_scope"] == "single_camera"
