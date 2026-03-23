from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json
import os

app = FastAPI()

# Data Model for the "Virtual Camera"
class CameraConfig(BaseModel):
    id: str
    name: str
    pos_x: float # Meters (0-10)
    pos_y: float # Meters (0-20)
    height: float # Meters
    tilt: float # Degrees (-90 to 0)
    pan: float # Degrees (0-360)
    focal_length: float # Zoom level

class CourtSetup(BaseModel):
    name: str
    cameras: List[CameraConfig]

# Mock Database (In-memory for now)
STORAGE_FILE = "court_setup.json"

@app.get("/setup")
def get_setup():
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, "r") as f:
            return json.load(f)
    return {"name": "Default Court", "cameras": []}

@app.post("/setup")
def save_setup(setup: CourtSetup):
    with open(STORAGE_FILE, "w") as f:
        json.dump(setup.dict(), f)
    return {"status": "success"}

@app.post("/analyze/start")
def start_analysis(video_path: str, setup_id: str):
    # This will trigger the PadelCV processing
    return {"status": "Analysis Started", "job_id": "123"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
