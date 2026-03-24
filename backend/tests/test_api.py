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
