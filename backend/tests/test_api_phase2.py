import pytest
from httpx import AsyncClient, ASGITransport
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from main import app


@pytest.fixture
def transport():
    return ASGITransport(app=app)


class TestScoreEndpoint:
    @pytest.mark.asyncio
    async def test_get_score_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/match/nonexistent/score")
            assert resp.status_code == 404


class TestEventsEndpoint:
    @pytest.mark.asyncio
    async def test_get_events_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/match/nonexistent/events")
            assert resp.status_code == 404


class TestTrajectoryEndpoint:
    @pytest.mark.asyncio
    async def test_get_trajectory_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/match/nonexistent/trajectory")
            assert resp.status_code == 404


class TestCorrectScore:
    @pytest.mark.asyncio
    async def test_correct_score_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/match/nonexistent/correct-score",
                                     json={"team": 1})
            assert resp.status_code == 404


class TestAssignPlayer:
    @pytest.mark.asyncio
    async def test_assign_player_no_match(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/match/nonexistent/assign-player",
                                     json={"track_id": 1, "player_id": "P1"})
            assert resp.status_code == 404


class TestAnalyzeStatus:
    @pytest.mark.asyncio
    async def test_analyze_status_no_job(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/analyze/status/nonexistent")
            assert resp.status_code == 404


class TestListMatches:
    @pytest.mark.asyncio
    async def test_list_matches_empty(self, transport):
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get("/matches")
            assert resp.status_code == 200
            data = resp.json()
            assert "matches" in data
