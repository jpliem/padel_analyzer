import pytest
import json
import os
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Create test client with isolated data directory."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import main
    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "matches"))
    return TestClient(main.app)


class TestHealthCheck:
    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


class TestMatchSetup:
    def test_create_match(self, client):
        payload = {
            "match_name": "Test Match",
            "players": {"P1": "Alice", "P2": "Bob", "P3": "Carol", "P4": "Dave"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "match_id" in data

    def test_get_match(self, client):
        payload = {
            "match_name": "Test Match",
            "players": {"P1": "Alice", "P2": "Bob", "P3": "Carol", "P4": "Dave"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        match_id = resp.json()["match_id"]
        resp = client.get(f"/match/{match_id}")
        assert resp.status_code == 200
        assert resp.json()["match_name"] == "Test Match"


class TestCalibration:
    def test_calibrate_court(self, client):
        payload = {
            "match_name": "Test",
            "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        match_id = resp.json()["match_id"]
        corners = {"corners": [[320, 700], [1600, 700], [1200, 200], [720, 200]]}
        resp = client.post(f"/match/{match_id}/calibrate", json=corners)
        assert resp.status_code == 200
        assert resp.json()["status"] == "calibrated"

    def test_calibrate_rejects_wrong_points(self, client):
        payload = {
            "match_name": "Test",
            "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        }
        resp = client.post("/match/setup", json=payload)
        match_id = resp.json()["match_id"]
        corners = {"corners": [[0, 0], [1, 1]]}
        resp = client.post(f"/match/{match_id}/calibrate", json=corners)
        assert resp.status_code == 400


SETUP_PAYLOAD = {
    "match_name": "Test",
    "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
    "teams": {"team_a": ["P1", "P2"], "team_b": ["P3", "P4"]},
}


class TestMultiCameraEndpoints:
    def _create_match(self, client):
        resp = client.post("/match/setup", json=SETUP_PAYLOAD)
        assert resp.status_code == 200
        return resp.json()["match_id"]

    def test_create_match_initializes_cameras_and_overrides(self, client):
        match_id = self._create_match(client)
        resp = client.get(f"/match/{match_id}")
        data = resp.json()
        assert data["cameras"] == []
        assert data["court_model_overrides"] is None

    def test_add_camera_to_match(self, client):
        match_id = self._create_match(client)
        resp = client.post(f"/match/{match_id}/cameras", json={
            "camera_id": "cam1",
            "label": "Center High",
            "source_type": "rtsp",
            "source_path": "rtsp://192.168.1.100/stream",
        })
        assert resp.status_code == 200
        assert resp.json()["camera_id"] == "cam1"
        assert resp.json()["status"] == "added"

    def test_add_camera_persists_in_match(self, client):
        match_id = self._create_match(client)
        client.post(f"/match/{match_id}/cameras", json={
            "camera_id": "cam1",
            "label": "Back",
            "source_type": "file",
            "source_path": "/videos/back.mp4",
        })
        resp = client.get(f"/match/{match_id}")
        cameras = resp.json()["cameras"]
        assert len(cameras) == 1
        assert cameras[0]["camera_id"] == "cam1"
        assert cameras[0]["label"] == "Back"
        assert cameras[0]["source_type"] == "file"

    def test_add_multiple_cameras(self, client):
        match_id = self._create_match(client)
        client.post(f"/match/{match_id}/cameras", json={"camera_id": "cam1", "label": "Left"})
        client.post(f"/match/{match_id}/cameras", json={"camera_id": "cam2", "label": "Right"})
        resp = client.get(f"/match/{match_id}")
        cameras = resp.json()["cameras"]
        assert len(cameras) == 2
        assert cameras[0]["camera_id"] == "cam1"
        assert cameras[1]["camera_id"] == "cam2"

    def test_add_camera_to_nonexistent_match(self, client):
        resp = client.post("/match/doesnotexist/cameras", json={"camera_id": "cam1"})
        assert resp.status_code == 404

    def test_calibrate_camera(self, client):
        match_id = self._create_match(client)
        client.post(f"/match/{match_id}/cameras", json={"camera_id": "cam1"})
        resp = client.post(f"/match/{match_id}/cameras/cam1/calibrate", json={
            "corners": [[320, 700], [1600, 700], [1200, 200], [720, 200]],
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "calibrated"
        assert resp.json()["camera_id"] == "cam1"

    def test_calibrate_camera_persists_data(self, client):
        match_id = self._create_match(client)
        client.post(f"/match/{match_id}/cameras", json={"camera_id": "cam1"})
        client.post(f"/match/{match_id}/cameras/cam1/calibrate", json={
            "corners": [[320, 700], [1600, 700], [1200, 200], [720, 200]],
        })
        resp = client.get(f"/match/{match_id}")
        cameras = resp.json()["cameras"]
        assert cameras[0]["camera_id"] == "cam1"
        assert "calibration" in cameras[0]
        assert "calibration_points" in cameras[0]

    def test_calibrate_nonexistent_camera(self, client):
        match_id = self._create_match(client)
        resp = client.post(f"/match/{match_id}/cameras/missing_cam/calibrate", json={
            "corners": [[320, 700], [1600, 700], [1200, 200], [720, 200]],
        })
        assert resp.status_code == 404

    def test_calibrate_camera_rejects_too_few_points(self, client):
        match_id = self._create_match(client)
        client.post(f"/match/{match_id}/cameras", json={"camera_id": "cam1"})
        resp = client.post(f"/match/{match_id}/cameras/cam1/calibrate", json={
            "corners": [[0, 0], [1, 1]],
        })
        assert resp.status_code == 400

    def test_set_court_model_overrides(self, client):
        match_id = self._create_match(client)
        resp = client.post(f"/match/{match_id}/court-model", json={
            "back_wall_height": 3.5,
            "side_glass_height": 2.5,
        })
        assert resp.status_code == 200
        assert resp.json()["status"] == "updated"

    def test_court_model_overrides_persisted(self, client):
        match_id = self._create_match(client)
        client.post(f"/match/{match_id}/court-model", json={
            "back_wall_height": 3.5,
            "side_glass_height": 2.5,
        })
        resp = client.get(f"/match/{match_id}")
        data = resp.json()
        assert data["court_model_overrides"]["back_wall_height"] == 3.5
        assert data["court_model_overrides"]["side_glass_height"] == 2.5

    def test_court_model_overrides_partial(self, client):
        """Only specified fields are non-null; unspecified fields are None."""
        match_id = self._create_match(client)
        client.post(f"/match/{match_id}/court-model", json={"back_wall_height": 4.0})
        resp = client.get(f"/match/{match_id}")
        overrides = resp.json()["court_model_overrides"]
        assert overrides["back_wall_height"] == 4.0
        assert overrides["side_glass_height"] is None

    def test_court_model_overrides_nonexistent_match(self, client):
        resp = client.post("/match/doesnotexist/court-model", json={"back_wall_height": 3.5})
        assert resp.status_code == 404


class TestHardening:
    """Production hardening: identifier validation, deletion semantics."""

    def _create_match(self, client):
        resp = client.post("/match/setup", json={
            "match_name": "Hardening Match",
            "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
        })
        return resp.json()["match_id"]

    def test_match_id_with_dotdot_rejected(self, client):
        resp = client.get("/match/has..dots")
        assert resp.status_code == 400

    def test_match_id_with_slash_encoded_rejected(self, client):
        resp = client.get("/match/%2e%2e%2fescape")
        assert resp.status_code in (400, 404)

    def test_template_id_traversal_rejected(self, client):
        resp = client.get("/templates/..%2fconfig")
        assert resp.status_code in (400, 404)

    def test_delete_missing_match_returns_404(self, client):
        resp = client.delete("/match/nonexistent0")
        assert resp.status_code == 404

    def test_delete_existing_match(self, client):
        match_id = self._create_match(client)
        resp = client.delete(f"/match/{match_id}")
        assert resp.status_code == 200
        assert client.get(f"/match/{match_id}").status_code == 404


class TestShutdownCancelsAnalyses:
    def test_shutdown_sets_cancel_on_active_analyzers(self, tmp_path, monkeypatch):
        """Server shutdown must request cancellation of running analyses so the
        process can exit gracefully instead of hanging on the analysis thread."""
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import main

        class DummyAnalyzer:
            cancel_requested = False

        dummy = DummyAnalyzer()
        monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "matches"))
        main._active_analyzers["shutdown-test"] = dummy
        try:
            with TestClient(main.app):
                pass  # exiting the context runs the shutdown event
        finally:
            main._active_analyzers.pop("shutdown-test", None)
        assert dummy.cancel_requested is True


class TestCalibrationQuality:
    def _create_match(self, client):
        resp = client.post("/match/setup", json={
            "match_name": "Reproj Test",
            "players": {"P1": "A", "P2": "B", "P3": "C", "P4": "D"},
            "teams": {"1": ["P1", "P2"], "2": ["P3", "P4"]},
            "golden_point": True,
            "format": "best_of_3",
        })
        return resp.json()["match_id"]

    def test_calibrate_returns_reprojection_error(self, client):
        match_id = self._create_match(client)
        keypoints = [
            [100, 680], [1180, 680], [280, 500], [640, 500], [1000, 500],
            [240, 400], [1040, 400], [300, 310], [640, 310], [980, 310],
            [380, 160], [900, 160],
        ]
        resp = client.post(f"/match/{match_id}/calibrate", json={
            "corners": keypoints,
            "net_top_points": [[230, 370], [1050, 370]],
            "image_width": 1280, "image_height": 720,
        })
        assert resp.status_code == 200
        body = resp.json()
        assert "reprojection_error" in body
        if body["mode"] == "3d":
            assert body["reprojection_error"] is not None
            assert 0.0 <= body["reprojection_error"] < 200.0

    def test_calibrate_four_corners_reports_null_error(self, client):
        match_id = self._create_match(client)
        resp = client.post(f"/match/{match_id}/calibrate", json={
            "corners": [[320, 700], [1600, 700], [1200, 200], [720, 200]],
        })
        assert resp.status_code == 200
        assert "reprojection_error" in resp.json()
