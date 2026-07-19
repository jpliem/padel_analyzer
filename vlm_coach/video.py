from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class ActivitySample:
    timestamp: float
    score: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start


def probe_video(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise ValueError("The uploaded file is not a readable video")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if fps <= 0 or frames <= 0:
        raise ValueError("Video duration could not be determined")
    return {
        "fps": round(fps, 3),
        "frames": frames,
        "duration": round(frames / fps, 3),
        "width": width,
        "height": height,
    }


def sample_activity(path: Path, sample_fps: float = 2.0) -> List[ActivitySample]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("The uploaded file is not a readable video")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    step = max(1, int(round(fps / sample_fps)))
    previous = None
    samples: List[ActivitySample] = []
    frame_number = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_number % step == 0:
            height, width = frame.shape[:2]
            target_width = 192
            target_height = max(1, int(height * target_width / max(width, 1)))
            small = cv2.resize(frame, (target_width, target_height))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            score = 0.0 if previous is None else float(np.mean(cv2.absdiff(gray, previous)))
            samples.append(ActivitySample(frame_number / fps, score))
            previous = gray
        frame_number += 1
    capture.release()
    return samples


def segments_from_activity(samples: Sequence[ActivitySample], duration: float,
                           threshold: float | None = None,
                           idle_gap: float = 3.0, min_duration: float = 4.0,
                           max_duration: float = 35.0,
                           padding: float = 1.5) -> List[Segment]:
    if duration <= 0:
        return []
    if not samples:
        return _fixed_segments(duration, max_duration)
    scores = np.array([sample.score for sample in samples], dtype=np.float32)
    if threshold is None:
        positive = scores[scores > 0]
        if positive.size:
            threshold = max(2.0, float(np.percentile(positive, 42)) * 0.65)
        else:
            threshold = 2.0
    active_times = [sample.timestamp for sample in samples if sample.score >= threshold]
    if not active_times:
        return _fixed_segments(duration, max_duration)

    raw: List[Segment] = []
    start = active_times[0]
    previous = start
    for timestamp in active_times[1:]:
        if timestamp - previous > idle_gap:
            raw.append(Segment(max(0.0, start - padding), min(duration, previous + padding)))
            start = timestamp
        previous = timestamp
    raw.append(Segment(max(0.0, start - padding), min(duration, previous + padding)))

    useful = [segment for segment in raw if segment.duration >= min_duration]
    if not useful:
        return _fixed_segments(duration, max_duration)

    split: List[Segment] = []
    for segment in useful:
        count = max(1, math.ceil(segment.duration / max_duration))
        part = segment.duration / count
        for index in range(count):
            split.append(Segment(segment.start + index * part,
                                 segment.start + (index + 1) * part))
    return split


def _fixed_segments(duration: float, max_duration: float) -> List[Segment]:
    segments = []
    start = 0.0
    while start < duration:
        end = min(duration, start + max_duration)
        if end - start >= 1.0:
            segments.append(Segment(start, end))
        start = end
    return segments


def detect_segments(path: Path) -> List[Segment]:
    media = probe_video(path)
    segments = segments_from_activity(sample_activity(path), media["duration"])
    # Keep local inference bounded on long recordings. Adjacent activity
    # windows are combined instead of silently dropping parts of the match.
    max_storyboards = 60
    while len(segments) > max_storyboards:
        merged: List[Segment] = []
        for index in range(0, len(segments), 2):
            if index + 1 < len(segments):
                merged.append(Segment(segments[index].start, segments[index + 1].end))
            else:
                merged.append(segments[index])
        segments = merged
    return segments


def extract_storyboard(path: Path, segment: Segment, output_dir: Path,
                       frame_count: int = 8, max_width: int = 768,
                       sampling: str = "uniform",
                       annotate_timeline: bool = False) -> List[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("Video could not be opened for storyboard extraction")
    if sampling == "scoring":
        timestamps = scoring_timestamps(segment, frame_count)
    elif sampling == "gap":
        safe_end = max(segment.start, segment.end - 0.05)
        timestamps = np.linspace(segment.start, safe_end, frame_count)
    elif frame_count <= 1:
        timestamps = [(segment.start + segment.end) / 2]
    else:
        # The exact container duration is often one frame past the last
        # decodable timestamp. Keep the final sample inside the segment.
        timestamps = np.linspace(segment.start, segment.end, frame_count, endpoint=False)
    frames = []
    for source_index, timestamp in enumerate(timestamps):
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        height, width = frame.shape[:2]
        if width > max_width:
            resized_height = max(1, int(height * max_width / width))
            frame = cv2.resize(frame, (max_width, resized_height), interpolation=cv2.INTER_AREA)
        # Multi-image VLM APIs preserve list order, but a visible label makes
        # chronology unambiguous to the model and survives processor changes.
        index = len(frames)
        if annotate_timeline:
            label = f"FRAME {index:02d}  TIME {float(timestamp):07.2f}s"
            cv2.rectangle(frame, (8, 8), (360, 45), (0, 0, 0), -1)
            cv2.putText(
                frame, label, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                (255, 255, 255), 2, cv2.LINE_AA,
            )
        destination = output_dir / f"frame_{index:02d}.jpg"
        cv2.imwrite(str(destination), frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        frames.append({
            "index": index,
            "source_index": source_index,
            "timestamp": round(float(timestamp), 3),
            "path": destination,
        })
    capture.release()
    if not frames:
        raise ValueError("No storyboard frames could be extracted")
    return frames


def scoring_timestamps(segment: Segment, frame_count: int) -> List[float]:
    """Cover the rally while reserving frames for its likely ending/reset."""
    if frame_count <= 1:
        return [(segment.start + segment.end) / 2]
    safe_end = max(segment.start, segment.end - 0.05)
    ending_count = max(3, math.ceil(frame_count * 0.4))
    broad_count = max(1, frame_count - ending_count)
    broad = np.linspace(segment.start, safe_end, broad_count, endpoint=False)
    ending_start = max(segment.start, safe_end - min(4.0, segment.duration * 0.45))
    ending = np.linspace(ending_start, safe_end, ending_count)
    combined = sorted({round(float(value), 4) for value in np.concatenate((broad, ending))})
    return combined


def storyboard_panels(frames: Sequence[dict], output_dir: Path,
                      rows: int = 2, columns: int = 2,
                      panel_width: int = 768) -> List[Path]:
    """Pack chronological frames into comic-style panels, row-major."""
    if rows <= 0 or columns <= 0 or panel_width <= 0:
        raise ValueError("panel dimensions must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    per_panel = rows * columns
    cell_width = panel_width // columns
    paths = []
    for panel_index, offset in enumerate(range(0, len(frames), per_panel)):
        images = []
        for item in frames[offset:offset + per_panel]:
            image = cv2.imread(str(item["path"]))
            if image is None:
                continue
            height, width = image.shape[:2]
            cell_height = max(1, int(height * cell_width / width))
            images.append(cv2.resize(image, (cell_width, cell_height), interpolation=cv2.INTER_AREA))
        if not images:
            continue
        cell_height = images[0].shape[0]
        canvas = np.zeros((rows * cell_height, columns * cell_width, 3), dtype=np.uint8)
        for local_index, image in enumerate(images):
            row, column = divmod(local_index, columns)
            canvas[row * cell_height:(row + 1) * cell_height,
                   column * cell_width:(column + 1) * cell_width] = image
        destination = output_dir / f"panel_{panel_index:02d}.jpg"
        cv2.imwrite(str(destination), canvas, [cv2.IMWRITE_JPEG_QUALITY, 86])
        paths.append(destination)
    return paths
