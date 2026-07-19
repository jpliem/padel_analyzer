from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline import MatchPipeline
from .provider import build_provider
from .store import MatchStore
from .video import probe_video


PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("VLM_COACH_DATA", PACKAGE_DIR / "data"))
STATIC_DIR = PACKAGE_DIR / "static"

store = MatchStore(DATA_DIR / "matches")
provider = build_provider()
# Backward-compatible name for tests and any early local integrations.
ollama = provider
pipeline = MatchPipeline(store, provider)
app = FastAPI(title="Padel Match Coach", version="0.1.0")
_running: set[str] = set()
_running_lock = threading.Lock()

# A local process cannot resume an in-flight Ollama request after a restart.
# Make that state visible and retryable instead of polling forever.
for _record in store.list():
    if _record.get("status") == "analyzing":
        store.update(
            _record["id"], status="error", stage="Analysis interrupted",
            error="The app stopped during analysis. Start the analysis again to continue.",
        )


class RallyReviewRequest(BaseModel):
    winner: Literal["team_a", "team_b", "unknown"] = "unknown"
    ending: Literal[
        "winner", "forced_error", "unforced_error", "net", "out",
        "double_bounce", "unknown",
    ] = "unknown"
    note: str = ""


def _public_record(record: dict, compact: bool = False) -> dict:
    result = dict(record)
    result["video_url"] = f"/api/matches/{record['id']}/recording"
    if compact:
        result.pop("rallies", None)
        result.pop("story", None)
    return result


@app.get("/api/health")
def health():
    state = provider.health()
    return {"status": "ok", "provider": state, "ollama": state,
            "service": "padel-match-coach"}


@app.get("/api/matches")
def list_matches():
    return {"matches": [_public_record(record, compact=True) for record in store.list()]}


@app.post("/api/matches")
def create_match(
    file: UploadFile = File(...),
    name: str = Form("Padel match"),
    team_a: str = Form("Team A"),
    team_b: str = Form("Team B"),
    model: str = Form("qwen3.5:2b"),
):
    if not file.filename:
        raise HTTPException(400, "Choose a video recording")
    record = store.create(name, team_a, team_b, model, file.filename)
    destination = store.video_path(record["id"])
    try:
        with destination.open("wb") as output:
            shutil.copyfileobj(file.file, output)
        media = probe_video(destination)
        record = store.update(record["id"], media=media)
    except Exception as exc:
        shutil.rmtree(store.directory(record["id"]), ignore_errors=True)
        raise HTTPException(400, str(exc)) from exc
    return _public_record(record)


@app.get("/api/matches/{match_id}")
def get_match(match_id: str):
    try:
        return _public_record(store.load(match_id))
    except KeyError:
        raise HTTPException(404, "Match not found")


@app.get("/api/matches/{match_id}/recording")
def get_recording(match_id: str):
    try:
        record = store.load(match_id)
    except KeyError:
        raise HTTPException(404, "Match not found")
    path = store.video_path(match_id)
    if not path.exists():
        raise HTTPException(404, "Recording not found")
    return FileResponse(path, media_type="video/mp4", filename=record.get("original_name", "recording.mp4"),
                        content_disposition_type="inline")


def _run_analysis(match_id: str) -> None:
    def progress(percent: int, stage: str):
        store.update(match_id, progress=percent, stage=stage, status="analyzing", error=None)

    try:
        pipeline.analyze(match_id, progress)
    except Exception as exc:
        store.update(match_id, status="error", stage="Analysis stopped", error=str(exc))
    finally:
        with _running_lock:
            _running.discard(match_id)


@app.post("/api/matches/{match_id}/analyze")
def analyze_match(match_id: str, background_tasks: BackgroundTasks):
    try:
        record = store.load(match_id)
    except KeyError:
        raise HTTPException(404, "Match not found")
    state = provider.health()
    if not state["available"]:
        raise HTTPException(503, state.get("error") or "Local VLM runtime is unavailable")
    installed = {name.split(":latest")[0] for name in state.get("models", []) if name}
    requested = record["model"]
    if requested not in state.get("models", []) and requested.split(":")[0] not in installed:
        raise HTTPException(409, f"Model '{requested}' is not installed. Run: ollama pull {requested}")
    with _running_lock:
        if match_id in _running:
            return {"status": "already_running"}
        _running.add(match_id)
    store.update(match_id, status="analyzing", progress=1, stage="Starting local Qwen", error=None)
    background_tasks.add_task(_run_analysis, match_id)
    return {"status": "started"}


@app.patch("/api/matches/{match_id}/rallies/{rally_id}")
def review_rally(match_id: str, rally_id: int, request: RallyReviewRequest):
    try:
        record = store.load(match_id)
    except KeyError:
        raise HTTPException(404, "Match not found")
    rally = next((item for item in record.get("rallies", []) if item["id"] == rally_id), None)
    if rally is None:
        raise HTTPException(404, "Rally not found")
    rally["review"] = request.model_dump() if hasattr(request, "model_dump") else request.dict()
    store.save(record)
    return {"rally": rally}


@app.post("/api/matches/{match_id}/story")
def regenerate_story(match_id: str):
    try:
        record = store.load(match_id)
    except KeyError:
        raise HTTPException(404, "Match not found")
    if not record.get("rallies"):
        raise HTTPException(409, "Analyze the match first")
    try:
        record["story"] = pipeline.build_story(record)
        store.save(record)
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"story": record["story"]}


@app.delete("/api/matches/{match_id}")
def delete_match(match_id: str):
    with _running_lock:
        if match_id in _running:
            raise HTTPException(409, "Wait for analysis to finish before deleting this match")
    try:
        directory = store.directory(match_id)
        store.load(match_id)
    except KeyError:
        raise HTTPException(404, "Match not found")
    shutil.rmtree(directory)
    return {"status": "deleted"}


app.mount("/media", StaticFiles(directory=DATA_DIR / "matches", check_dir=False), name="media")


@app.get("/{path:path}")
def frontend(path: str = ""):
    requested = STATIC_DIR / path
    if path and requested.is_file() and STATIC_DIR in requested.resolve().parents:
        return FileResponse(requested)
    return FileResponse(STATIC_DIR / "index.html")


def main() -> None:
    port = int(os.environ.get("VLM_COACH_PORT", "8765"))
    uvicorn.run("vlm_coach.app:app", host="127.0.0.1", port=port, reload=False)


if __name__ == "__main__":
    main()
