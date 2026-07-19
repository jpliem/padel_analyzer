from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


class MatchStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def create(self, name: str, team_a: str, team_b: str, model: str,
               original_name: str) -> Dict:
        match_id = uuid.uuid4().hex[:10]
        now = datetime.now(timezone.utc).isoformat()
        record = {
            "id": match_id,
            "name": name.strip() or "Padel match",
            "team_a": team_a.strip() or "Team A",
            "team_b": team_b.strip() or "Team B",
            "model": model.strip() or "qwen3.5:2b",
            "original_name": original_name,
            "created_at": now,
            "updated_at": now,
            "status": "uploaded",
            "progress": 0,
            "stage": "Ready to analyze",
            "error": None,
            "media": None,
            "rallies": [],
            "story": None,
            "performance": None,
        }
        self.save(record)
        return record

    def directory(self, match_id: str) -> Path:
        if not match_id or any(c not in "0123456789abcdef" for c in match_id):
            raise KeyError(match_id)
        return self.root / match_id

    def video_path(self, match_id: str) -> Path:
        return self.directory(match_id) / "recording.mp4"

    def load(self, match_id: str) -> Dict:
        path = self.directory(match_id) / "match.json"
        if not path.exists():
            raise KeyError(match_id)
        with self._lock, path.open() as handle:
            return json.load(handle)

    def save(self, record: Dict) -> None:
        record = dict(record)
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        directory = self.directory(record["id"])
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "match.json"
        with self._lock:
            fd, temporary = tempfile.mkstemp(prefix="match-", suffix=".json", dir=directory)
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(record, handle, indent=2)
                os.replace(temporary, destination)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)

    def update(self, match_id: str, **changes) -> Dict:
        with self._lock:
            record = self.load(match_id)
            record.update(changes)
            self.save(record)
            return record

    def list(self) -> List[Dict]:
        records = []
        for path in self.root.glob("*/match.json"):
            try:
                with path.open() as handle:
                    records.append(json.load(handle))
            except (OSError, ValueError):
                continue
        records.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return records
