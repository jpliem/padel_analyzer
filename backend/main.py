import sys
import os

# Ensure src/ is on Python path for consistent imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
import logging
import re
import uuid
import shutil
import asyncio
import csv
import time
import io
import subprocess
from datetime import datetime, timezone

logging.basicConfig(level=os.environ.get("PADEL_LOG_LEVEL", "INFO"))
logger = logging.getLogger("padel_analyzer")

app = FastAPI(title="Padel Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+)(:\d+)?$",
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/matches"


@app.on_event("shutdown")
def _cancel_active_analyses() -> None:
    """Request cancellation of running analyses so shutdown does not hang on
    the analysis thread (a plain SIGTERM previously required kill -9)."""
    for match_id, analyzer in list(_active_analyzers.items()):
        if getattr(analyzer, "cancel_requested", None) is False:
            analyzer.cancel_requested = True
            logger.info("Shutdown: requested cancel for analysis %s", match_id)


class MatchSetupRequest(BaseModel):
    match_name: str
    players: Dict[str, str]
    teams: Dict[str, List[str]]
    golden_point: bool = True
    format: str = "best_of_3"
    out_of_court_play_enabled: bool = False
    first_server: Optional[str] = None


class CalibrationRequest(BaseModel):
    corners: List[List[float]]  # 4 corners (legacy) or 12 keypoints
    net_points: Optional[List[List[float]]] = None  # 2 net post ground points
    net_top_points: Optional[List[List[float]]] = None  # 2 net post TOP points (for 3D)
    image_width: Optional[int] = 1280
    image_height: Optional[int] = 720


class AddCameraRequest(BaseModel):
    camera_id: str
    label: str = ""
    source_type: str = "file"  # "file" or "rtsp"
    source_path: str = ""


class CourtModelOverrideRequest(BaseModel):
    back_wall_height: Optional[float] = None
    side_glass_height: Optional[float] = None
    side_glass_depth: Optional[float] = None
    side_fence_height: Optional[float] = None
    out_of_court_play_enabled: Optional[bool] = None
    gate_center_y: Optional[float] = None
    gate_width: Optional[float] = None
    safety_zone_lateral_depth: Optional[float] = None
    safety_zone_y_margin: Optional[float] = None


_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
MAX_UPLOAD_BYTES = int(os.environ.get("PADEL_MAX_UPLOAD_BYTES", str(8 * 1024 ** 3)))


def _require_safe_id(value: str, kind: str = "id") -> str:
    """Reject identifiers that could escape their storage directory."""
    if not value or not _SAFE_ID_RE.match(value) or ".." in value:
        raise HTTPException(status_code=400, detail=f"Invalid {kind}")
    return value


def _match_dir(match_id: str) -> str:
    return os.path.join(DATA_DIR, _require_safe_id(match_id, "match id"))


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


def _write_json(path: str, data: dict) -> None:
    """Write a JSON document atomically so an interrupted process cannot corrupt it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp"
    with open(temporary, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _probe_video(path: str, original_name: str = "video.mp4") -> dict:
    """Read browser-facing media facts from the uploaded recording."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable video")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {
        "original_name": original_name,
        "fps": round(fps, 3),
        "frame_count": frames,
        "duration_seconds": round(frames / fps, 3) if fps > 0 else 0.0,
        "width": width,
        "height": height,
        "size_bytes": os.path.getsize(path),
        "uploaded_at": _utc_now(),
    }


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
            if config.get("media"):
                status = "uploaded"
            if config.get("calibration"):
                status = "calibrated"
            results_path = os.path.join(DATA_DIR, match_id, "results.json")
            job = _load_job(match_id)
            if job and job.get("state") == "processing":
                status = "processing"
            if os.path.exists(results_path):
                status = "analyzed"
            matches.append({
                "match_id": match_id,
                "match_name": config.get("match_name", "Unknown"),
                "status": status,
                "media": config.get("media"),
                "created_at": config.get("created_at"),
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
        "out_of_court_play_enabled": req.out_of_court_play_enabled,
        "first_server": req.first_server,
        "calibration": None,
        "cameras": [],
        "court_model_overrides": None,
        "media": None,
        "created_at": _utc_now(),
    }
    _save_match(match_id, data)
    return {"match_id": match_id, "status": "created"}


@app.get("/match/{match_id}")
def get_match(match_id: str):
    return _load_match(match_id)


@app.post("/match/{match_id}/cameras")
def add_camera(match_id: str, req: AddCameraRequest):
    match_data = _load_match(match_id)
    camera_entry = {
        "camera_id": req.camera_id,
        "label": req.label,
        "source_type": req.source_type,
        "source_path": req.source_path,
    }
    cameras = match_data.get("cameras", [])
    cameras.append(camera_entry)
    match_data["cameras"] = cameras
    _save_match(match_id, match_data)
    return {"camera_id": req.camera_id, "status": "added"}


@app.post("/match/{match_id}/cameras/{cam_id}/calibrate")
def calibrate_camera(match_id: str, cam_id: str, req: CalibrationRequest):
    import numpy as np
    from cv.court_calibration import CourtCalibration
    from cv.camera_model import CameraModel

    match_data = _load_match(match_id)
    cameras = match_data.get("cameras", [])
    camera = next((c for c in cameras if c["camera_id"] == cam_id), None)
    if camera is None:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found in match")

    n_points = len(req.corners)
    if n_points < 4:
        raise HTTPException(status_code=400, detail=f"Need at least 4 keypoints, got {n_points}")

    cal = CourtCalibration()
    if n_points >= 12:
        cal.calibrate_keypoints(req.corners)
    else:
        net = np.array(req.net_points, dtype=np.float32) if req.net_points else None
        cal.calibrate(np.array(req.corners, dtype=np.float32), net_pixels=net)

    cam = CameraModel()
    cam.calibrate(
        keypoints_2d=req.corners,
        net_top_2d=req.net_top_points,
        image_width=req.image_width or 1280,
        image_height=req.image_height or 720,
    )

    mode = "3d" if cam.has_3d() else ("12-keypoint" if n_points >= 12 else "4-corner")

    camera["calibration"] = cal.to_dict()
    camera["camera_model"] = cam.to_dict()
    camera["calibration_points"] = {
        "corners": req.corners,
        "net_points": req.net_points,
        "net_top_points": req.net_top_points,
        "mode": mode,
    }
    _save_match(match_id, match_data)
    return {"status": "calibrated", "camera_id": cam_id}


@app.post("/match/{match_id}/court-model")
def set_court_model_overrides(match_id: str, req: CourtModelOverrideRequest):
    match_data = _load_match(match_id)
    match_data["court_model_overrides"] = req.model_dump(exclude_none=False)
    _save_match(match_id, match_data)
    return {"status": "updated"}


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
    reprojection_error = cam.compute_reprojection_error(req.corners)

    match_data = _load_match(match_id)
    match_data["calibration"] = cal.to_dict()
    match_data["camera_model"] = cam.to_dict()
    match_data["calibration_points"] = {
        "corners": req.corners,
        "net_points": req.net_points,
        "net_top_points": req.net_top_points,
        "mode": mode,
    }
    match_data["reprojection_error"] = reprojection_error
    _save_match(match_id, match_data)
    return {"status": "calibrated", "match_id": match_id, "mode": mode,
            "reprojection_error": reprojection_error}


@app.delete("/match/{match_id}/analysis")
def delete_analysis(match_id: str):
    """Delete analysis results so match can be re-analyzed."""
    _load_match(match_id)
    match_dir = _match_dir(match_id)
    for fname in ["results.json", "annotated.mp4", "analysis_status.json"]:
        path = os.path.join(match_dir, fname)
        if os.path.exists(path):
            os.remove(path)
    highlight_dir = os.path.join(match_dir, "highlights")
    if os.path.isdir(highlight_dir):
        shutil.rmtree(highlight_dir)
    _active_analyzers.pop(match_id, None)
    _analysis_jobs.pop(match_id, None)
    return {"status": "deleted", "match_id": match_id}


@app.delete("/match/{match_id}")
def delete_match(match_id: str):
    """Delete entire match and all its data."""
    match_dir = _match_dir(match_id)
    if not os.path.exists(match_dir):
        raise HTTPException(status_code=404, detail="Match not found")
    shutil.rmtree(match_dir)
    _active_analyzers.pop(match_id, None)
    _analysis_jobs.pop(match_id, None)
    return {"status": "deleted", "match_id": match_id}


# -- Calibration Templates --

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
    path = os.path.join(TEMPLATES_DIR, f"{_require_safe_id(template_id, 'template id')}.json")
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
    path = os.path.join(TEMPLATES_DIR, f"{_require_safe_id(template_id, 'template id')}.json")
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
    path = os.path.join(TEMPLATES_DIR, f"{_require_safe_id(template_id, 'template id')}.json")
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


def _save_results(match_id: str, data: dict) -> None:
    data["updated_at"] = _utc_now()
    _write_json(os.path.join(_match_dir(match_id), "results.json"), data)


def _job_path(match_id: str) -> str:
    return os.path.join(_match_dir(match_id), "analysis_status.json")


def _save_job(match_id: str, job: dict) -> None:
    job["match_id"] = match_id
    job["updated_at"] = _utc_now()
    _analysis_jobs[match_id] = job
    _write_json(_job_path(match_id), job)


def _load_job(match_id: str) -> dict | None:
    if match_id in _analysis_jobs:
        return _analysis_jobs[match_id]
    path = _job_path(match_id)
    if os.path.exists(path):
        with open(path) as f:
            job = json.load(f)
        if job.get("state") == "processing":
            job = {
                **job, "state": "error",
                "error": "Analysis was interrupted by a server restart. Start it again to continue.",
            }
            _write_json(path, job)
        return job
    if _load_results(match_id) is not None:
        return {"state": "complete", "percent": 100, "match_id": match_id}
    return None


def _event_json(event) -> dict:
    return {
        "event_type": event.event_type.value,
        "timestamp": event.timestamp,
        "frame_number": event.frame_number,
        "position": {"x": event.position.x, "y": event.position.y},
        "confidence": getattr(event, "confidence", 0.0),
        "metadata": event.metadata,
    }


def _build_highlights(events: list, media: dict) -> list:
    """Create seekable rally segments without inventing events the model did not see."""
    duration = float(media.get("duration_seconds", 0.0) or 0.0)
    ordered = sorted(events, key=lambda event: float(event.get("timestamp", 0.0)))
    highlights = []
    rally_start = None
    for event in ordered:
        timestamp = max(0.0, float(event.get("timestamp", 0.0)))
        kind = event.get("event_type")
        if kind == "SERVE":
            rally_start = timestamp
        elif rally_start is None and kind in ("HIT", "BOUNCE"):
            rally_start = timestamp
        if kind == "POINT_END":
            start = max(0.0, (rally_start if rally_start is not None else timestamp - 8.0) - 2.0)
            end = timestamp + 3.0
            if duration:
                end = min(duration, end)
            highlights.append({
                "id": f"rally-{len(highlights) + 1}",
                "title": f"Rally {len(highlights) + 1}",
                "start_seconds": round(start, 3),
                "end_seconds": round(max(start, end), 3),
                "duration_seconds": round(max(0.0, end - start), 3),
                "end_reason": event.get("metadata", {}).get("reason", "point ended"),
                "confidence": float(event.get("confidence", 0.0) or 0.0),
                "needs_review": float(event.get("confidence", 0.0) or 0.0) < 0.85,
            })
            rally_start = None
    return highlights


def _build_stats(results: dict) -> dict:
    events = results.get("events", [])
    highlights = results.get("highlights", [])
    counts: Dict[str, int] = {}
    for event in events:
        kind = event.get("event_type", "UNKNOWN")
        counts[kind] = counts.get(kind, 0) + 1
    rally_durations = [float(item.get("duration_seconds", 0.0)) for item in highlights]
    reviews = results.get("reviews", [])
    return {
        "rallies": len(highlights),
        "total_events": len(events),
        "hits": counts.get("HIT", 0),
        "bounces": counts.get("BOUNCE", 0),
        "wall_hits": counts.get("WALL_HIT", 0),
        "serves": counts.get("SERVE", 0),
        "faults": counts.get("FAULT", 0),
        "average_rally_seconds": round(sum(rally_durations) / len(rally_durations), 2) if rally_durations else 0.0,
        "longest_rally_seconds": round(max(rally_durations), 2) if rally_durations else 0.0,
        "ball_track_points": len(results.get("trajectory", [])),
        "frames_processed": int(results.get("frames_processed", 0)),
        "pending_reviews": sum(1 for review in reviews if review.get("status") == "proposed"),
    }


def _upgrade_saved_results(match_id: str, saved: dict, match_data: dict) -> dict:
    """Fill product fields missing from analyses created by older releases."""
    changed = False

    defaults = {
        "score": {"score": "0 - 0", "games": "0 - 0", "sets": "0 - 0"},
        "events": [], "trajectory": [], "player_positions": [], "reviews": [],
        "wall_hits": [], "frames_processed": 0, "model_scope": "single_camera",
        "accuracy_notice": "Ball depth, occlusions, glass contacts, and close line calls are estimates and may require review.",
        "model_info": None, "active_ball_diagnostics": {},
        "evidence_status": {}, "contact_proposals": [],
        "semantic_observations": [], "rule_decisions": [],
        "system_scope": {},
    }
    for key, value in defaults.items():
        if saved.get(key) is None:
            saved[key] = value
            changed = True

    if not saved.get("media"):
        media = match_data.get("media")
        video_path = os.path.join(_match_dir(match_id), "video.mp4")
        if not media and os.path.exists(video_path):
            try:
                media = _probe_video(video_path)
            except HTTPException:
                media = None
        saved["media"] = media or {
            "original_name": "Legacy recording", "fps": 30.0,
            "frame_count": saved.get("frames_processed", 0), "duration_seconds": 0.0,
            "width": 0, "height": 0, "size_bytes": 0, "uploaded_at": "",
        }
        changed = True

    if saved.get("highlights") is None:
        saved["highlights"] = _build_highlights(saved["events"], saved["media"])
        changed = True
    if saved.get("stats") is None:
        saved["stats"] = _build_stats(saved)
        changed = True
    if changed:
        _save_results(match_id, saved)
    return saved


def _analysis_payload(analyzer, result: dict, media: dict) -> dict:
    events = [_event_json(event) for event in analyzer.all_events]
    payload = {
        "score": analyzer.scoring_engine.get_score_display(),
        "events": events,
        "wall_hits": [event for event in events if event["event_type"] == "WALL_HIT"],
        "trajectory": analyzer.ball_tracker.trajectory,
        # 8–10 Hz is smooth enough for the court map and keeps long matches usable.
        "player_positions": analyzer.player_positions_log[::3],
        "frames_processed": result.get("frames_processed", 0),
        "reviews": [_record_json(record) for record in analyzer.review_ledger.records],
        "media": media,
        "model_scope": "single_camera",
        "accuracy_notice": "Ball depth, occlusions, glass contacts, and close line calls are estimates and may require review.",
        "model_info": result.get("model_info", analyzer.model_info),
        "active_ball_diagnostics": result.get(
            "active_ball_diagnostics", analyzer.active_ball_selector.diagnostics()),
        "evidence_status": result.get("evidence_status", {}),
        "contact_proposals": result.get("contact_proposals", []),
        "semantic_observations": result.get("semantic_observations", []),
        "rule_decisions": result.get("rule_decisions", []),
        "system_scope": {
            "runtime": [
                "single-camera ball candidates and active-ball temporal gating",
                "player tracking and re-identification",
                "court homography and cautious monocular trajectory estimates",
                "optional audio impulse evidence",
                "padel serve/rally rules with a durable human review ledger",
            ],
            "research_only": [
                "CalTennis multi-camera triangulation demo",
                "tennis and table-tennis dataset experiments",
                "VLM ball probing",
            ],
            "not_validated": [
                "automatic scoring accuracy",
                "net-touch recognition from the current single view",
                "reliable depth during every occlusion",
                "generalization across clubs, lighting, cameras, and court colors",
            ],
        },
    }
    payload["highlights"] = _build_highlights(events, media)
    payload["stats"] = _build_stats(payload)
    return payload


def _restore_annotated_audio(original_path: str, annotated_path: str) -> None:
    """Mux source audio into the overlay video when ffmpeg is available."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not os.path.exists(annotated_path):
        return
    muxed = f"{annotated_path}.muxed.mp4"
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-i", annotated_path, "-i", original_path,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
        "-shortest", "-movflags", "+faststart", muxed,
    ]
    try:
        completed = subprocess.run(command, capture_output=True, timeout=180, check=False)
        if completed.returncode == 0 and os.path.getsize(muxed) > 0:
            os.replace(muxed, annotated_path)
    except (OSError, subprocess.SubprocessError):
        # Audio is an enhancement; never discard an otherwise valid analysis.
        return
    finally:
        if os.path.exists(muxed):
            os.remove(muxed)


def _replay_saved_score(match_id: str, reviews: list) -> dict:
    """Rebuild scoring from the durable point ledger in chronological order."""
    from logic.scoring_engine import PadelScoringEngine
    from models.types import PointReason

    config = _build_match_config(_load_match(match_id))
    engine = PadelScoringEngine(
        golden_point=config.golden_point,
        sets_to_win=config.format.value,
        first_server=config.first_server,
        team_players=config.teams,
    )
    indexed = list(enumerate(reviews))
    for _, record in sorted(indexed, key=lambda item: (item[1].get("frame_number", 0), item[0])):
        if record.get("status") == "confirmed" and record.get("winner_team") in (1, 2):
            try:
                reason = PointReason(record.get("reason", "manual"))
            except ValueError:
                reason = PointReason.MANUAL
            engine.add_point(record["winner_team"], reason)
    return engine.get_score_display()


def _persist_active_review_state(match_id: str, analyzer) -> dict:
    score = analyzer.refresh_score_from_reviews()
    saved = _load_results(match_id)
    if saved is not None:
        saved["reviews"] = [_record_json(record) for record in analyzer.review_ledger.records]
        saved["score"] = score
        saved["stats"] = _build_stats(saved)
        _save_results(match_id, saved)
    return score


class CorrectScoreRequest(BaseModel):
    team: int


class ResolveReviewRequest(BaseModel):
    confirmed: bool
    winner_team: Optional[int] = None
    note: str = ""


class CorrectReviewRequest(BaseModel):
    winner_team: int
    reason: str = "manual"
    note: str = "manual correction"


class ProposeReviewRequest(BaseModel):
    frame_number: int
    winner_team: Optional[int] = None
    reason: str = "manual"
    confidence: float = 0.0
    source: str = "vlm"


def _build_match_config(match_data: dict):
    from models.types import MatchConfig, MatchFormat, ServerInfo, TeamId

    teams = {}
    for key, players in match_data.get("teams", {}).items():
        normalised = str(key).lower()
        team = TeamId.TEAM_A if normalised in ("1", "team_a", "a") else TeamId.TEAM_B
        teams[team] = players
    config = MatchConfig(
        match_name=match_data.get("match_name", "Match"),
        players=match_data.get("players", {}),
        teams=teams or MatchConfig().teams,
        golden_point=match_data.get("golden_point", True),
        format=(MatchFormat.BEST_OF_1 if match_data.get("format") == "best_of_1"
                else MatchFormat.BEST_OF_3),
        out_of_court_play_enabled=match_data.get("out_of_court_play_enabled", False),
    )
    first_player = match_data.get("first_server")
    if first_player:
        team = next((team for team, players in config.teams.items()
                     if first_player in players), TeamId.TEAM_A)
        config.first_server = ServerInfo(team_id=team, player_id=first_player)
    return config


def _record_json(record):
    return {
        "id": record.id, "frame_number": record.frame_number,
        "winner_team": record.winner_team, "reason": record.reason.value,
        "confidence": record.confidence, "source": record.source,
        "status": record.status.value, "supersedes": record.supersedes,
        "note": record.note,
    }


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
    analyzer = VideoAnalyzer(
        match_id=match_id, calibration=cal, config=config,
        match_config=_build_match_config(match_data), detector_type=detector_type,
        court_model_overrides=match_data.get("court_model_overrides"),
    )
    _active_analyzers[match_id] = analyzer
    _save_job(match_id, {"state": "processing", "percent": 0, "detector_type": detector_type})

    def progress_cb(frame, total, pct):
        job = _analysis_jobs[match_id]
        new_percent = round(pct, 1)
        if int(new_percent) != int(job.get("percent", -1)):
            _save_job(match_id, {**job, "state": "processing", "percent": new_percent})

    def run():
        try:
            annotated_path = os.path.join(_match_dir(match_id), "annotated.mp4")
            result = analyzer.analyze_video(video_path, progress_callback=progress_cb,
                                            annotated_path=annotated_path)
            if result.get("cancelled"):
                _save_job(match_id, {"state": "cancelled",
                                     "percent": _analysis_jobs[match_id].get("percent", 0)})
                return
            _restore_annotated_audio(video_path, annotated_path)
            media = match_data.get("media") or _probe_video(video_path)
            _save_results(match_id, _analysis_payload(analyzer, result, media))
            _save_job(match_id, {"state": "complete", "percent": 100})
        except Exception:
            logger.exception("Analysis failed for match %s", match_id)
            _save_job(match_id, {
                "state": "error",
                "percent": _analysis_jobs[match_id].get("percent", 0),
                "error": "Analysis failed. Check server logs for details.",
            })

    background_tasks.add_task(run)
    return {"status": "started", "match_id": match_id}


@app.post("/match/{match_id}/auto-detect-court")
def auto_detect_court(match_id: str):
    """Run court keypoint detection on first frame of uploaded video."""
    _load_match(match_id)
    video_path = os.path.join(_match_dir(match_id), "video.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=400, detail="No video uploaded")

    import cv2
    from cv.court_detector import CourtDetector

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(status_code=400, detail="Cannot read video frame")

    detector = CourtDetector()
    keypoints = detector.detect(frame)
    if not keypoints:
        raise HTTPException(status_code=400, detail="Could not detect court keypoints")

    # Auto-calibrate with detected keypoints
    import numpy as np
    from cv.court_calibration import CourtCalibration
    from cv.camera_model import CameraModel

    cal = CourtCalibration()
    cal.calibrate_keypoints(keypoints)

    cam = CameraModel()
    cam.calibrate(keypoints_2d=keypoints,
                  image_width=frame.shape[1], image_height=frame.shape[0])

    match_data = _load_match(match_id)
    match_data["calibration"] = cal.to_dict()
    match_data["camera_model"] = cam.to_dict()
    match_data["calibration_points"] = {
        "corners": keypoints,
        "mode": "auto-detect",
    }
    _save_match(match_id, match_data)

    return {
        "status": "calibrated",
        "mode": "auto-detect" + (" (3D)" if cam.has_3d() else ""),
        "keypoints": keypoints,
    }


@app.get("/match/{match_id}/annotated")
def get_annotated_video(match_id: str):
    _load_match(match_id)
    video_path = os.path.join(_match_dir(match_id), "annotated.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Annotated video not available")
    return FileResponse(video_path, media_type="video/mp4", filename="annotated.mp4",
                        content_disposition_type="inline")


@app.get("/match/{match_id}/recording")
def get_original_video(match_id: str):
    match_data = _load_match(match_id)
    video_path = os.path.join(_match_dir(match_id), "video.mp4")
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="Recording not available")
    media = match_data.get("media") or {}
    return FileResponse(
        video_path,
        media_type=media.get("content_type", "video/mp4"),
        filename=media.get("original_name", "recording.mp4"),
        content_disposition_type="inline",
    )


@app.get("/match/{match_id}/result")
def get_match_result(match_id: str):
    match_data = _load_match(match_id)
    saved = _load_results(match_id)
    if saved is not None:
        saved = _upgrade_saved_results(match_id, saved, match_data)
    return {
        "match": match_data,
        "analysis": saved,
        "job": _load_job(match_id) or {"state": "not_started", "percent": 0},
    }


@app.get("/match/{match_id}/highlights")
def get_highlights(match_id: str):
    match_data = _load_match(match_id)
    saved = _load_results(match_id) or {}
    highlights = saved.get("highlights")
    if highlights is None:
        highlights = _build_highlights(saved.get("events", []), saved.get("media") or match_data.get("media") or {})
    return {"highlights": highlights}


@app.get("/match/{match_id}/highlights/{highlight_id}.mp4")
def download_highlight_clip(match_id: str, highlight_id: str):
    """Render and cache a shareable clip from a detected rally segment."""
    _load_match(match_id)
    saved = _load_results(match_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Analysis results not available")
    highlight = next((item for item in saved.get("highlights", [])
                      if item.get("id") == highlight_id), None)
    if highlight is None:
        raise HTTPException(status_code=404, detail="Highlight not found")
    source = os.path.join(_match_dir(match_id), "video.mp4")
    if not os.path.exists(source):
        raise HTTPException(status_code=404, detail="Recording not available")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise HTTPException(status_code=503, detail="ffmpeg is required to create clips")
    clip_dir = os.path.join(_match_dir(match_id), "highlights")
    os.makedirs(clip_dir, exist_ok=True)
    clip_path = os.path.join(clip_dir, f"{highlight_id}.mp4")
    if not os.path.exists(clip_path):
        command = [
            ffmpeg, "-y", "-loglevel", "error",
            "-ss", str(highlight["start_seconds"]), "-i", source,
            "-t", str(max(0.1, highlight["duration_seconds"])),
            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
            "-movflags", "+faststart", clip_path,
        ]
        completed = subprocess.run(command, capture_output=True, timeout=300, check=False)
        if completed.returncode != 0 or not os.path.exists(clip_path):
            raise HTTPException(status_code=500, detail="Could not create highlight clip")
    return FileResponse(clip_path, media_type="video/mp4",
                        filename=f"{highlight_id}.mp4")


@app.get("/match/{match_id}/export.json")
def export_match_json(match_id: str):
    match_data = _load_match(match_id)
    saved = _load_results(match_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Analysis results not available")
    return JSONResponse(
        {"match": match_data, "analysis": saved},
        headers={"Content-Disposition": f'attachment; filename="{match_id}-analysis.json"'},
    )


@app.get("/match/{match_id}/export.csv")
def export_match_csv(match_id: str):
    _load_match(match_id)
    saved = _load_results(match_id)
    if saved is None:
        raise HTTPException(status_code=404, detail="Analysis results not available")
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["timestamp_seconds", "frame", "event", "x", "y", "confidence", "details"])
    for event in saved.get("events", []):
        position = event.get("position", {})
        writer.writerow([
            event.get("timestamp", 0), event.get("frame_number", 0), event.get("event_type", ""),
            position.get("x", ""), position.get("y", ""), event.get("confidence", ""),
            json.dumps(event.get("metadata", {}), separators=(",", ":")),
        ])
    return Response(
        content=output.getvalue(), media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{match_id}-events.csv"'},
    )


@app.get("/match/{match_id}/stats")
def get_stats(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        saved = _load_results(match_id)
        return {"stats": saved.get("stats", _build_stats(saved)) if saved else {}}
    return {"stats": {
        "total_events": len(analyzer.all_events),
        "frames_processed": analyzer._frame_count if hasattr(analyzer, '_frame_count') else 0,
    }}


@app.post("/match/{match_id}/correct-score")
def correct_score(match_id: str, req: CorrectScoreRequest):
    _load_match(match_id)
    if req.team not in (1, 2):
        raise HTTPException(status_code=400, detail="team must be 1 or 2")
    analyzer = _active_analyzers.get(match_id)
    if analyzer is not None:
        record = analyzer.add_manual_point(req.team)
        score = _persist_active_review_state(match_id, analyzer)
        return {"status": "corrected", "score": score, "review": _record_json(record)}
    saved = _load_results(match_id)
    if saved is None:
        raise HTTPException(status_code=400, detail="Analyze the recording before correcting its score")
    reviews = saved.setdefault("reviews", [])
    record = {
        "id": str(uuid.uuid4()), "frame_number": saved.get("frames_processed", 0),
        "winner_team": req.team, "reason": "manual", "confidence": 1.0,
        "source": "manual", "status": "confirmed", "supersedes": None,
        "note": "manual point",
    }
    reviews.append(record)
    saved["score"] = _replay_saved_score(match_id, reviews)
    saved["stats"] = _build_stats(saved)
    _save_results(match_id, saved)
    return {"status": "corrected", "score": saved["score"], "review": record}


@app.get("/match/{match_id}/reviews")
def get_reviews(match_id: str):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        saved = _load_results(match_id) or {}
        return {"reviews": saved.get("reviews", [])}
    return {"reviews": [_record_json(r) for r in analyzer.review_ledger.records]}


@app.post("/match/{match_id}/reviews/propose")
def propose_review(match_id: str, req: ProposeReviewRequest):
    from models.types import PointReason

    _load_match(match_id)
    if req.source not in ("vlm", "vision", "audio", "manual"):
        raise HTTPException(status_code=400, detail="unsupported review source")
    try:
        reason = PointReason(req.reason)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        saved = _load_results(match_id)
        if saved is None:
            raise HTTPException(status_code=400, detail="Analyze the recording before adding a review")
        record = {
            "id": str(uuid.uuid4()), "frame_number": req.frame_number,
            "winner_team": req.winner_team, "reason": reason.value,
            "confidence": min(max(req.confidence, 0.0), 1.0), "source": req.source,
            "status": "proposed", "supersedes": None, "note": "",
        }
        saved.setdefault("reviews", []).append(record)
        saved["stats"] = _build_stats(saved)
        _save_results(match_id, saved)
        return {"review": record}
    # External model proposals always require a person, regardless of the confidence.
    record = analyzer.review_ledger.propose(
        req.frame_number, req.winner_team, reason,
        min(max(req.confidence, 0.0), 1.0), req.source,
        auto_confirm_threshold=2.0,
    )
    return {"review": _record_json(record)}


@app.post("/match/{match_id}/reviews/{record_id}/resolve")
def resolve_review(match_id: str, record_id: str, req: ResolveReviewRequest):
    _load_match(match_id)
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        saved = _load_results(match_id)
        if saved is None:
            raise HTTPException(status_code=400, detail="Analysis results not available")
        record = next((item for item in saved.get("reviews", []) if item.get("id") == record_id), None)
        if record is None:
            raise HTTPException(status_code=400, detail="Review record not found")
        winner = record.get("winner_team") if req.winner_team is None else req.winner_team
        if req.confirmed and winner not in (1, 2):
            raise HTTPException(status_code=400, detail="a confirmed point needs a winner")
        record.update({"winner_team": winner, "note": req.note,
                       "status": "confirmed" if req.confirmed else "rejected"})
        saved["score"] = _replay_saved_score(match_id, saved.get("reviews", []))
        saved["stats"] = _build_stats(saved)
        _save_results(match_id, saved)
        return {"review": record, "score": saved["score"]}
    try:
        record = analyzer.review_ledger.resolve(
            record_id, confirmed=req.confirmed,
            winner_team=req.winner_team, note=req.note)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    score = _persist_active_review_state(match_id, analyzer)
    return {"review": _record_json(record), "score": score}


@app.post("/match/{match_id}/reviews/{record_id}/correct")
def correct_review(match_id: str, record_id: str, req: CorrectReviewRequest):
    from models.types import PointReason

    _load_match(match_id)
    if req.winner_team not in (1, 2):
        raise HTTPException(status_code=400, detail="winner_team must be 1 or 2")
    analyzer = _active_analyzers.get(match_id)
    if analyzer is None:
        saved = _load_results(match_id)
        if saved is None:
            raise HTTPException(status_code=400, detail="Analysis results not available")
        old = next((item for item in saved.get("reviews", []) if item.get("id") == record_id), None)
        if old is None:
            raise HTTPException(status_code=400, detail="Review record not found")
        old["status"] = "superseded"
        record = {
            "id": str(uuid.uuid4()), "frame_number": old.get("frame_number", 0),
            "winner_team": req.winner_team, "reason": req.reason,
            "confidence": 1.0, "source": "manual", "status": "confirmed",
            "supersedes": old["id"], "note": req.note,
        }
        saved.setdefault("reviews", []).append(record)
        saved["score"] = _replay_saved_score(match_id, saved["reviews"])
        saved["stats"] = _build_stats(saved)
        _save_results(match_id, saved)
        return {"review": record, "score": saved["score"]}
    try:
        reason = PointReason(req.reason)
        record = analyzer.review_ledger.correct(
            record_id, req.winner_team, reason, req.note)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    score = _persist_active_review_state(match_id, analyzer)
    return {"review": _record_json(record), "score": score}


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
                       detector_type: str = "tracknet"):
    match_data = _load_match(match_id)
    if file.content_type and not file.content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a video file")
    match_dir = _match_dir(match_id)
    os.makedirs(match_dir, exist_ok=True)
    video_path = os.path.join(match_dir, "video.mp4")
    written = 0
    with open(video_path, "wb") as f:
        while chunk := file.file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                f.close()
                os.remove(video_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Video exceeds the {MAX_UPLOAD_BYTES // (1024 ** 3)} GB upload limit",
                )
            f.write(chunk)
    job_id = match_id
    try:
        media = _probe_video(video_path, file.filename or "video.mp4")
    except Exception:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise
    media["content_type"] = file.content_type or "video/mp4"
    match_data["media"] = media
    _save_match(match_id, match_data)
    _save_job(job_id, {
        "state": "uploaded", "percent": 0,
        "detector_type": detector_type,
    })
    return {"job_id": job_id, "status": "uploaded", "media": media}


@app.post("/analyze/start/{job_id}")
def start_analysis(job_id: str, background_tasks: BackgroundTasks):
    persisted_job = _load_job(job_id)
    if persisted_job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    job = persisted_job
    match_id = job["match_id"]
    match_data = _load_match(match_id)

    import numpy as np
    from cv.court_calibration import CourtCalibration
    from models.config import EventDetectorConfig
    from pipeline.video_analyzer import VideoAnalyzer

    cal = _load_calibration(match_data)

    config = EventDetectorConfig()
    detector_type = job.get("detector_type", "tracknet")
    analyzer = VideoAnalyzer(
        match_id=match_id, calibration=cal, config=config,
        match_config=_build_match_config(match_data), detector_type=detector_type,
        court_model_overrides=match_data.get("court_model_overrides"),
    )
    _active_analyzers[match_id] = analyzer

    video_path = os.path.join(_match_dir(match_id), "video.mp4")

    analysis_started_at = time.monotonic()

    def progress_cb(frame, total, pct):
        current = _analysis_jobs[job_id]
        new_percent = round(pct, 1)
        if int(new_percent) != int(current.get("percent", -1)):
            _save_job(job_id, {**current, "percent": new_percent, "state": "processing"})
            elapsed = time.monotonic() - analysis_started_at
            if pct > 0:
                eta_min = elapsed * (100.0 - pct) / pct / 60.0
                logger.info("Analysis %s: %.0f%% (frame %d/%d, ETA %.0f min)",
                            job_id, pct, frame, total, eta_min)

    def run_analysis():
        _save_job(job_id, {**job, "state": "processing", "percent": 0})
        try:
            annotated_path = os.path.join(_match_dir(match_id), "annotated.mp4")
            result = analyzer.analyze_video(video_path, progress_callback=progress_cb,
                                            annotated_path=annotated_path)
            if result.get("cancelled"):
                _save_job(job_id, {"state": "cancelled",
                                   "percent": _analysis_jobs[job_id].get("percent", 0)})
                return
            _restore_annotated_audio(video_path, annotated_path)
            media = match_data.get("media") or _probe_video(video_path)
            _save_results(match_id, _analysis_payload(analyzer, result, media))
            _save_job(job_id, {"state": "complete", "percent": 100})
        except Exception:
            logger.exception("Analysis failed for job %s", job_id)
            _save_job(job_id, {
                "state": "error",
                "percent": _analysis_jobs[job_id].get("percent", 0),
                "error": "Analysis failed. Check server logs for details.",
            })

    background_tasks.add_task(run_analysis)
    return {"status": "started", "job_id": job_id}


@app.post("/analyze/cancel/{job_id}")
def cancel_analysis(job_id: str):
    """Request cancellation of a running analysis. Stops within one frame."""
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    analyzer = _active_analyzers.get(job.get("match_id", job_id))
    if analyzer is None or job.get("state") not in ("processing", "uploaded"):
        raise HTTPException(status_code=409, detail="No running analysis to cancel")
    analyzer.cancel_requested = True
    _save_job(job_id, {**_analysis_jobs.get(job_id, job), "state": "cancelling"})
    return {"status": "cancelling", "job_id": job_id}


@app.get("/analyze/status/{job_id}")
def get_analysis_status(job_id: str):
    job = _load_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


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
    analyzer = VideoAnalyzer(
        match_id=req.match_id, calibration=cal, config=config,
        match_config=_build_match_config(match_data), detector_type="yolo",
        court_model_overrides=match_data.get("court_model_overrides"),
    )
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
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
