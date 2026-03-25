import sys
import os

# Ensure src/ is on Python path for consistent imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import uuid
import shutil
import asyncio

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
    corners: List[List[float]]  # 4 corners (legacy) or 12 keypoints
    net_points: Optional[List[List[float]]] = None  # 2 net post ground points
    net_top_points: Optional[List[List[float]]] = None  # 2 net post TOP points (for 3D)
    image_width: Optional[int] = 1280
    image_height: Optional[int] = 720


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


def _load_calibration(match_data: dict):
    """Load the best available calibration — CameraModel (3D) or CourtCalibration (2D)."""
    from cv.court_calibration import CourtCalibration
    from cv.camera_model import CameraModel

    # Prefer CameraModel if available
    if match_data.get("camera_model"):
        cam = CameraModel.from_dict(match_data["camera_model"])
        if cam.has_3d():
            return cam

    # Fallback to CourtCalibration
    if match_data.get("calibration"):
        return CourtCalibration.from_dict(match_data["calibration"])

    return CourtCalibration()


@app.get("/")
def root():
    return {"status": "ok", "service": "padel-analyzer"}


@app.get("/matches")
def list_matches():
    if not os.path.exists(DATA_DIR):
        return {"matches": []}
    matches = []
    for match_id in os.listdir(DATA_DIR):
        config_path = os.path.join(DATA_DIR, match_id, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                config = json.load(f)
            status = "created"
            if config.get("calibration"):
                status = "calibrated"
            results_path = os.path.join(DATA_DIR, match_id, "results.json")
            if os.path.exists(results_path) or match_id in _active_analyzers:
                status = "analyzed"
            matches.append({
                "match_id": match_id,
                "match_name": config.get("match_name", "Unknown"),
                "status": status,
            })
    return {"matches": matches}


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
    import numpy as np
    from cv.court_calibration import CourtCalibration
    from cv.camera_model import CameraModel

    n_points = len(req.corners)
    if n_points < 4:
        raise HTTPException(status_code=400, detail=f"Need at least 4 keypoints, got {n_points}")

    # Legacy CourtCalibration (for backward compat)
    cal = CourtCalibration()
    if n_points >= 12:
        cal.calibrate_keypoints(req.corners)
    else:
        net = np.array(req.net_points, dtype=np.float32) if req.net_points else None
        cal.calibrate(np.array(req.corners, dtype=np.float32), net_pixels=net)

    # New: 3D CameraModel (for accurate projection)
    cam = CameraModel()
    cam.calibrate(
        keypoints_2d=req.corners,
        net_top_2d=req.net_top_points,
        image_width=req.image_width or 1280,
        image_height=req.image_height or 720,
    )

    mode = "3d" if cam.has_3d() else ("12-keypoint" if n_points >= 12 else "4-corner")

    match_data = _load_match(match_id)
    match_data["calibration"] = cal.to_dict()
    match_data["camera_model"] = cam.to_dict()
    match_data["calibration_points"] = {
        "corners": req.corners,
        "net_points": req.net_points,
        "net_top_points": req.net_top_points,
        "mode": mode,
    }
    _save_match(match_id, match_data)
    return {"status": "calibrated", "match_id": match_id, "mode": mode}


@app.delete("/match/{match_id}/analysis")
def delete_analysis(match_id: str):
    """Delete analysis results so match can be re-analyzed."""
    _load_match(match_id)
    match_dir = _match_dir(match_id)
    for fname in ["results.json", "annotated.mp4"]:
        path = os.path.join(match_dir, fname)
        if os.path.exists(path):
            os.remove(path)
    _active_analyzers.pop(match_id, None)
    _analysis_jobs.pop(match_id, None)
    return {"status": "deleted", "match_id": match_id}


@app.delete("/match/{match_id}")
def delete_match(match_id: str):
    """Delete entire match and all its data."""
    match_dir = _match_dir(match_id)
    if os.path.exists(match_dir):
        shutil.rmtree(match_dir)
    _active_analyzers.pop(match_id, None)
    _analysis_jobs.pop(match_id, None)
    return {"status": "deleted", "match_id": match_id}


# ── Calibration Templates ──────────────────────────────────────────────

TEMPLATES_DIR = "data/templates"


class SaveTemplateRequest(BaseModel):
    name: str
    corners: List[List[float]]
    net_points: Optional[List[List[float]]] = None
    thumbnail: Optional[str] = None  # base64 JPEG


@app.get("/templates")
def list_templates():
    if not os.path.exists(TEMPLATES_DIR):
        return {"templates": []}
    templates = []
    for fname in os.listdir(TEMPLATES_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(TEMPLATES_DIR, fname)) as f:
                t = json.load(f)
                templates.append({
                    "id": fname.replace(".json", ""),
                    "name": t.get("name", "Unknown"),
                    "has_thumbnail": t.get("thumbnail") is not None,
                })
    return {"templates": templates}


@app.get("/templates/{template_id}")
def get_template(template_id: str):
    path = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Template not found")
    with open(path) as f:
        return json.load(f)


@app.post("/templates")
def save_template(req: SaveTemplateRequest):
    os.makedirs(TEMPLATES_DIR, exist_ok=True)
    template_id = str(uuid.uuid4())[:8]
    data = {
        "id": template_id,
        "name": req.name,
        "corners": req.corners,
        "net_points": req.net_points,
        "thumbnail": req.thumbnail,
    }
    with open(os.path.join(TEMPLATES_DIR, f"{template_id}.json"), "w") as f:
        json.dump(data, f, indent=2)
    return {"id": template_id, "status": "saved"}


@app.get("/templates/{template_id}/thumbnail")
def get_template_thumbnail(template_id: str):
    path = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Template not found")
    with open(path) as f:
        data = json.load(f)
    if not data.get("thumbnail"):
        raise HTTPException(status_code=404, detail="No thumbnail")
    import base64
    img_bytes = base64.b64decode(data["thumbnail"])
    from fastapi.responses import Response
    return Response(content=img_bytes, media_type="image/jpeg")


@app.delete("/templates/{template_id}")
def delete_template(template_id: str):
    path = os.path.join(TEMPLATES_DIR, f"{template_id}.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Template not found")
    os.remove(path)
    return {"status": "deleted"}


_active_analyzers: Dict[str, object] = {}


def _load_results(match_id: str) -> dict | None:
    """Load saved analysis results from disk."""
    path = os.path.join(_match_dir(match_id), "results.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


class CorrectScoreRequest(BaseModel):
    team: int


class AssignPlayerRequest(BaseModel):
    track_id: int
    player_id: str


@app.get("/match/{match_id}/score")
def get_score(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is not None:
        return analyzer.scoring_engine.get_score_display()
    saved = _load_results(match_id)
    if saved and "score" in saved:
        return saved["score"]
    return {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"}


@app.get("/match/{match_id}/events")
def get_events(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is not None:
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
    saved = _load_results(match_id)
    if saved and "events" in saved:
        return {"events": saved["events"]}
    return {"events": []}


@app.get("/match/{match_id}/trajectory")
def get_trajectory(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is not None:
        return {"trajectory": analyzer.ball_tracker.trajectory}
    saved = _load_results(match_id)
    if saved and "trajectory" in saved:
        return {"trajectory": saved["trajectory"]}
    return {"trajectory": []}


@app.get("/match/{match_id}/positions")
def get_positions(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is not None:
        return {"positions": analyzer.player_positions_log}
    saved = _load_results(match_id)
    if saved and "player_positions" in saved:
        return {"positions": saved["player_positions"]}
    return {"positions": []}


@app.post("/match/{match_id}/analyze")
def start_match_analysis(match_id: str, background_tasks: BackgroundTasks,
                         detector_type: str = "tracknet"):
    """One-click analysis: checks if video exists on disk, starts processing."""
    match_data = _load_match(match_id)
    video_path = os.path.join(_match_dir(match_id), "video.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="No video uploaded for this match")

    import numpy as np
    from cv.court_calibration import CourtCalibration
    from models.config import EventDetectorConfig
    from pipeline.video_analyzer import VideoAnalyzer

    cal = _load_calibration(match_data)

    config = EventDetectorConfig()
    analyzer = VideoAnalyzer(match_id=match_id, calibration=cal, config=config,
                             detector_type=detector_type)
    _active_analyzers[match_id] = analyzer
    _analysis_jobs[match_id] = {"state": "processing", "percent": 0, "match_id": match_id}

    def progress_cb(frame, total, pct):
        _analysis_jobs[match_id]["percent"] = round(pct, 1)

    def run():
        try:
            annotated_path = os.path.join(_match_dir(match_id), "annotated.mp4")
            result = analyzer.analyze_video(video_path, progress_callback=progress_cb,
                                            annotated_path=annotated_path)
            _analysis_jobs[match_id]["state"] = "complete"
            _analysis_jobs[match_id]["percent"] = 100
            results_file = os.path.join(_match_dir(match_id), "results.json")
            with open(results_file, "w") as f:
                json.dump({
                    "score": analyzer.scoring_engine.get_score_display(),
                    "events": [
                        {"event_type": e.event_type.value, "timestamp": e.timestamp,
                         "frame_number": e.frame_number,
                         "position": {"x": e.position.x, "y": e.position.y},
                         "metadata": e.metadata}
                        for e in analyzer.all_events
                    ],
                    "trajectory": analyzer.ball_tracker.trajectory,
                    "player_positions": analyzer.player_positions_log,
                    "frames_processed": result.get("frames_processed", 0),
                }, f)
        except Exception as e:
            _analysis_jobs[match_id]["state"] = "error"
            _analysis_jobs[match_id]["error"] = str(e)

    background_tasks.add_task(run)
    return {"status": "started", "match_id": match_id}


@app.get("/match/{match_id}/annotated")
def get_annotated_video(match_id: str):
    _load_match(match_id)
    video_path = os.path.join(_match_dir(match_id), "annotated.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Annotated video not available")
    return FileResponse(video_path, media_type="video/mp4", filename="annotated.mp4")


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


_analysis_jobs: Dict[str, Dict] = {}


@app.post("/analyze/upload")
async def upload_video(match_id: str, file: UploadFile = File(...),
                       detector_type: str = "yolo"):
    _load_match(match_id)
    match_dir = _match_dir(match_id)
    os.makedirs(match_dir, exist_ok=True)
    video_path = os.path.join(match_dir, "video.mp4")
    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    job_id = match_id
    _analysis_jobs[job_id] = {
        "state": "uploaded", "percent": 0,
        "match_id": match_id, "detector_type": detector_type,
    }
    return {"job_id": job_id, "status": "uploaded"}


@app.post("/analyze/start/{job_id}")
def start_analysis(job_id: str, background_tasks: BackgroundTasks):
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    job = _analysis_jobs[job_id]
    match_id = job["match_id"]
    match_data = _load_match(match_id)

    import numpy as np
    from cv.court_calibration import CourtCalibration
    from models.config import EventDetectorConfig
    from pipeline.video_analyzer import VideoAnalyzer

    cal = _load_calibration(match_data)

    config = EventDetectorConfig()
    detector_type = job.get("detector_type", "yolo")
    analyzer = VideoAnalyzer(match_id=match_id, calibration=cal, config=config,
                             detector_type=detector_type)
    _active_analyzers[match_id] = analyzer

    video_path = os.path.join(_match_dir(match_id), "video.mp4")

    def progress_cb(frame, total, pct):
        _analysis_jobs[job_id]["percent"] = round(pct, 1)
        _analysis_jobs[job_id]["state"] = "processing"

    def run_analysis():
        _analysis_jobs[job_id]["state"] = "processing"
        try:
            annotated_path = os.path.join(_match_dir(match_id), "annotated.mp4")
            result = analyzer.analyze_video(video_path, progress_callback=progress_cb,
                                            annotated_path=annotated_path)
            _analysis_jobs[job_id]["state"] = "complete"
            _analysis_jobs[job_id]["percent"] = 100
            # Save results to disk so they persist across server restarts
            results_file = os.path.join(_match_dir(match_id), "results.json")
            with open(results_file, "w") as f:
                json.dump({
                    "score": analyzer.scoring_engine.get_score_display(),
                    "events": [
                        {
                            "event_type": e.event_type.value,
                            "timestamp": e.timestamp,
                            "frame_number": e.frame_number,
                            "position": {"x": e.position.x, "y": e.position.y},
                            "metadata": e.metadata,
                        }
                        for e in analyzer.all_events
                    ],
                    "trajectory": analyzer.ball_tracker.trajectory,
                    "player_positions": analyzer.player_positions_log,
                    "frames_processed": result.get("frames_processed", 0),
                }, f)
        except Exception as e:
            _analysis_jobs[job_id]["state"] = "error"
            _analysis_jobs[job_id]["error"] = str(e)

    background_tasks.add_task(run_analysis)
    return {"status": "started", "job_id": job_id}


@app.get("/analyze/status/{job_id}")
def get_analysis_status(job_id: str):
    if job_id not in _analysis_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _analysis_jobs[job_id]


class LiveStartRequest(BaseModel):
    device_id: int = 0
    rtsp_url: Optional[str] = None
    match_id: str
    record: bool = False


_live_manager = None


@app.post("/live/start")
def start_live(req: LiveStartRequest):
    global _live_manager
    match_data = _load_match(req.match_id)

    import numpy as np
    from cv.court_calibration import CourtCalibration
    from models.config import EventDetectorConfig
    from pipeline.video_analyzer import VideoAnalyzer
    from pipeline.live_manager import LiveManager

    cal = _load_calibration(match_data)

    config = EventDetectorConfig()
    # Use YOLO for live mode (faster than TrackNet, skip frames compensates)
    analyzer = VideoAnalyzer(match_id=req.match_id, calibration=cal, config=config,
                             detector_type="yolo")
    _active_analyzers[req.match_id] = analyzer

    device = req.rtsp_url if req.rtsp_url else req.device_id
    record_path = os.path.join(_match_dir(req.match_id), "recording.mp4") if req.record else None

    _live_manager = LiveManager(analyzer, device_id=device,
                                record=req.record, record_path=record_path)
    _live_manager.start()
    return {"status": "started", "match_id": req.match_id}


@app.post("/live/stop")
def stop_live():
    global _live_manager
    if _live_manager is None:
        raise HTTPException(status_code=400, detail="No live session running")
    _live_manager.stop()
    _live_manager = None
    return {"status": "stopped"}


@app.get("/live/replay")
def get_replay():
    if _live_manager is None:
        raise HTTPException(status_code=400, detail="No live session running")
    import base64
    frames = _live_manager.get_replay()
    return {"frames": [
        {"jpeg": base64.b64encode(f["jpeg"]).decode(), "timestamp": f["timestamp"]}
        for f in frames
    ]}


@app.websocket("/live/stream")
async def live_stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            if _live_manager and _live_manager.is_running:
                frame_b64 = _live_manager.get_latest_frame_b64()
                if frame_b64:
                    await ws.send_json({"type": "frame", "jpeg": frame_b64})

                try:
                    event = _live_manager._event_queue.get_nowait()
                    await ws.send_json(event)
                except asyncio.QueueEmpty:
                    pass

                try:
                    data = await asyncio.wait_for(ws.receive_json(), timeout=0.01)
                    if data.get("type") == "correct" and _live_manager._analyzer:
                        _live_manager._analyzer.scoring_engine.add_point(data["team"])
                    elif data.get("type") == "reassign" and _live_manager._analyzer:
                        _live_manager._analyzer.player_tracker.assign_player(
                            data["track_id"], data["player_id"])
                except asyncio.TimeoutError:
                    pass

                if _live_manager._latest_result:
                    await ws.send_json({
                        "type": "score",
                        "data": _live_manager._latest_result.score,
                    })

            await asyncio.sleep(1 / 30)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
