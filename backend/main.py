import sys
import os

# Ensure src/ is on Python path for consistent imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import uuid

app = FastAPI(title="Padel Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/matches"


class MatchSetupRequest(BaseModel):
    match_name: str
    players: Dict[str, str]
    teams: Dict[str, List[str]]
    golden_point: bool = True
    format: str = "best_of_3"


class CalibrationRequest(BaseModel):
    corners: List[List[float]]


def _match_dir(match_id: str) -> str:
    return os.path.join(DATA_DIR, match_id)


def _load_match(match_id: str) -> dict:
    path = os.path.join(_match_dir(match_id), "config.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Match not found")
    with open(path) as f:
        return json.load(f)


def _save_match(match_id: str, data: dict):
    d = _match_dir(match_id)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(data, f, indent=2)


@app.get("/")
def root():
    return {"status": "ok", "service": "padel-analyzer"}


@app.post("/match/setup")
def create_match(req: MatchSetupRequest):
    match_id = str(uuid.uuid4())[:8]
    data = {
        "match_id": match_id,
        "match_name": req.match_name,
        "players": req.players,
        "teams": req.teams,
        "golden_point": req.golden_point,
        "format": req.format,
        "calibration": None,
    }
    _save_match(match_id, data)
    return {"match_id": match_id, "status": "created"}


@app.get("/match/{match_id}")
def get_match(match_id: str):
    return _load_match(match_id)


@app.post("/match/{match_id}/calibrate")
def calibrate_court(match_id: str, req: CalibrationRequest):
    if len(req.corners) != 4:
        raise HTTPException(status_code=400, detail="Exactly 4 corner points required")
    import numpy as np
    from cv.court_calibration import CourtCalibration
    cal = CourtCalibration()
    cal.calibrate(np.array(req.corners, dtype=np.float32))
    match_data = _load_match(match_id)
    match_data["calibration"] = cal.to_dict()
    _save_match(match_id, match_data)
    return {"status": "calibrated", "match_id": match_id}


_active_analyzers: Dict[str, object] = {}


class CorrectScoreRequest(BaseModel):
    team: int


class AssignPlayerRequest(BaseModel):
    track_id: int
    player_id: str


@app.get("/match/{match_id}/score")
def get_score(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"}
    return analyzer.scoring_engine.get_score_display()


@app.get("/match/{match_id}/events")
def get_events(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"events": []}
    return {"events": [
        {
            "event_type": e.event_type.value,
            "timestamp": e.timestamp,
            "frame_number": e.frame_number,
            "position": {"x": e.position.x, "y": e.position.y},
            "metadata": e.metadata,
        }
        for e in analyzer.all_events
    ]}


@app.get("/match/{match_id}/trajectory")
def get_trajectory(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"trajectory": []}
    return {"trajectory": analyzer.ball_tracker.trajectory}


@app.get("/match/{match_id}/stats")
def get_stats(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        return {"stats": {}}
    return {"stats": {
        "total_events": len(analyzer.all_events),
        "frames_processed": analyzer._frame_count if hasattr(analyzer, '_frame_count') else 0,
    }}


@app.post("/match/{match_id}/correct-score")
def correct_score(match_id: str, req: CorrectScoreRequest):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        raise HTTPException(status_code=400, detail="No active analysis for this match")
    analyzer.scoring_engine.add_point(req.team)
    return {"status": "corrected", "score": analyzer.scoring_engine.get_score_display()}


@app.post("/match/{match_id}/assign-player")
def assign_player(match_id: str, req: AssignPlayerRequest):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        raise HTTPException(status_code=400, detail="No active analysis for this match")
    analyzer.player_tracker.assign_player(req.track_id, req.player_id)
    return {"status": "assigned", "track_id": req.track_id, "player_id": req.player_id}


# Legacy endpoint for backwards compatibility
@app.get("/setup")
def get_setup():
    storage = "court_setup.json"
    if os.path.exists(storage):
        with open(storage) as f:
            return json.load(f)
    return {"name": "Default Court", "cameras": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
