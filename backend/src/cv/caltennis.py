"""Timestamp-aware adapter for the public CalTennis multi-view dataset."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from cv.camera_model import CameraModel


_CAPTURE_TIME = re.compile(
    r"^(?P<month>\d{2})_(?P<day>\d{2})_(?P<year>\d{4})_"
    r"(?P<hour>\d{2})_(?P<minute>\d{2})_(?P<second>\d{2})_(?P<millis>\d{3})"
)


def capture_start_seconds(video_id: str) -> float:
    """Return the capture start as a Unix timestamp parsed from a video ID."""
    match = _CAPTURE_TIME.match(Path(video_id).stem)
    if not match:
        raise ValueError(f"CalTennis video ID has no capture timestamp: {video_id!r}")
    values = {key: int(value) for key, value in match.groupdict().items()}
    dt = datetime(
        values["year"], values["month"], values["day"], values["hour"],
        values["minute"], values["second"], values["millis"] * 1000,
    )
    return dt.timestamp()


@dataclass
class CalTennisStream:
    camera_id: str
    video_path: Path
    timestamps_path: Path
    calibration_path: Path
    start_offset_seconds: float
    timestamps_ms: np.ndarray
    camera: CameraModel
    calibration_reprojection_px: float

    @property
    def projection_matrix(self) -> np.ndarray:
        matrix = self.camera.projection_matrix()
        if matrix is None:
            raise RuntimeError(f"camera {self.camera_id!r} has no projection matrix")
        return matrix

    @property
    def session_interval(self) -> Tuple[float, float]:
        return (
            self.start_offset_seconds + float(self.timestamps_ms[0]) / 1000.0,
            self.start_offset_seconds + float(self.timestamps_ms[-1]) / 1000.0,
        )

    def frame_at_session_time(self, session_seconds: float) -> int:
        """Return the nearest timestamped frame for a shared session time."""
        local_ms = (float(session_seconds) - self.start_offset_seconds) * 1000.0
        index = int(np.searchsorted(self.timestamps_ms, local_ms, side="left"))
        if index <= 0:
            return 0
        if index >= len(self.timestamps_ms):
            return len(self.timestamps_ms) - 1
        before = self.timestamps_ms[index - 1]
        after = self.timestamps_ms[index]
        return index - 1 if abs(local_ms - before) <= abs(after - local_ms) else index

    def session_time_for_frame(self, frame: int) -> float:
        if frame < 0 or frame >= len(self.timestamps_ms):
            raise IndexError(f"frame {frame} outside timestamp array")
        return self.start_offset_seconds + float(self.timestamps_ms[frame]) / 1000.0


@dataclass
class CalTennisSession:
    session_id: str
    streams: List[CalTennisStream]

    @property
    def overlap_interval(self) -> Tuple[float, float]:
        if not self.streams:
            raise ValueError("session has no camera streams")
        starts, ends = zip(*(stream.session_interval for stream in self.streams))
        overlap = max(starts), min(ends)
        if overlap[1] <= overlap[0]:
            raise ValueError(f"session {self.session_id!r} camera streams do not overlap")
        return overlap


def _load_camera(path: Path) -> Tuple[CameraModel, float]:
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    camera = CameraModel.from_parameters(
        data["K"], data["R_w2c"], data["t_w2c"], data.get("dist_coeffs")
    )
    return camera, float(data.get("reprojection_error", float("nan")))


def load_session(dataset_root: Path | str, session_id: str,
                 metadata_name: str = "metadata_mini.jsonl") -> CalTennisSession:
    """Load every indexed view for one session from a CalTennis repository."""
    root = Path(dataset_root).expanduser().resolve()
    metadata_path = root / metadata_name
    rows = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("session_id") == session_id:
                rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"expected at least two indexed views for {session_id!r}")

    absolute_starts = [capture_start_seconds(row["video_id"]) for row in rows]
    session_start = min(absolute_starts)
    streams = []
    for row, absolute_start in zip(rows, absolute_starts):
        video_path = root / row["video"]
        timestamps_path = root / row["timestamps"]
        calibration_path = root / row["calibration"]
        for path in (video_path, timestamps_path, calibration_path):
            if not path.exists():
                raise FileNotFoundError(path)
        timestamps = np.asarray(np.load(timestamps_path), dtype=np.float64)
        if timestamps.ndim != 1 or len(timestamps) == 0:
            raise ValueError(f"invalid timestamp array: {timestamps_path}")
        if np.any(np.diff(timestamps) < 0):
            raise ValueError(f"timestamps are not monotonic: {timestamps_path}")
        camera, error = _load_camera(calibration_path)
        streams.append(CalTennisStream(
            camera_id=row["video_id"],
            video_path=video_path,
            timestamps_path=timestamps_path,
            calibration_path=calibration_path,
            start_offset_seconds=absolute_start - session_start,
            timestamps_ms=timestamps,
            camera=camera,
            calibration_reprojection_px=error,
        ))
    streams.sort(key=lambda stream: stream.start_offset_seconds)
    return CalTennisSession(session_id=session_id, streams=streams)


def read_frame(stream: CalTennisStream, frame: int):
    """Read one encoded video frame for a timestamp-array frame index."""
    capture = cv2.VideoCapture(str(stream.video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = capture.read()
        if not ok:
            raise RuntimeError(f"could not read frame {frame} from {stream.video_path}")
        return image
    finally:
        capture.release()
