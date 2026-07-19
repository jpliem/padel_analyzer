from fastapi.testclient import TestClient

from logic.review_ledger import ReviewLedger


class FakeAnalyzer:
    def __init__(self):
        self.review_ledger = ReviewLedger()
        self.scoring_engine = self.review_ledger.replay()

    def refresh_score_from_reviews(self):
        self.scoring_engine = self.review_ledger.replay()
        return self.scoring_engine.get_score_display()


def _setup(tmp_path, monkeypatch):
    import main

    monkeypatch.setattr(main, "DATA_DIR", str(tmp_path / "matches"))
    main._active_analyzers.clear()
    client = TestClient(main.app)
    response = client.post("/match/setup", json={
        "match_name": "Review", "players": {}, "teams": {},
        "golden_point": True, "format": "best_of_3",
    })
    match_id = response.json()["match_id"]
    main._active_analyzers[match_id] = FakeAnalyzer()
    return client, match_id


def test_vlm_proposal_never_auto_confirms(tmp_path, monkeypatch):
    client, match_id = _setup(tmp_path, monkeypatch)
    response = client.post(f"/match/{match_id}/reviews/propose", json={
        "frame_number": 42, "winner_team": 1, "reason": "out",
        "confidence": 1.0, "source": "vlm",
    })
    assert response.status_code == 200
    assert response.json()["review"]["status"] == "proposed"


def test_review_confirmation_replays_score(tmp_path, monkeypatch):
    client, match_id = _setup(tmp_path, monkeypatch)
    record = client.post(f"/match/{match_id}/reviews/propose", json={
        "frame_number": 42, "winner_team": 2, "reason": "out",
        "confidence": .8, "source": "vision",
    }).json()["review"]
    response = client.post(
        f"/match/{match_id}/reviews/{record['id']}/resolve",
        json={"confirmed": True},
    )
    assert response.json()["score"]["score"] == "0 - 15"
