"""Adaptive audio impulse detector for racket, floor, wall, and net candidates."""

from dataclasses import dataclass
import shutil
import subprocess
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class AudioImpulse:
    timestamp: float
    strength: float
    confidence: float


class AudioImpulseDetector:
    def __init__(self, sample_rate: int, window_ms: float = 5.0,
                 threshold_mad: float = 8.0, refractory_ms: float = 35.0):
        self.sample_rate = sample_rate
        self.window = max(1, int(sample_rate * window_ms / 1000.0))
        self.threshold_mad = threshold_mad
        self.refractory = max(1, int(sample_rate * refractory_ms / 1000.0))

    def detect(self, samples: np.ndarray, start_time: float = 0.0) -> List[AudioImpulse]:
        samples = np.asarray(samples, dtype=float).reshape(-1)
        if len(samples) < self.window * 3:
            return []
        # Short-window energy works for impact transients and is robust to sign.
        kernel = np.ones(self.window) / self.window
        energy = np.convolve(samples * samples, kernel, mode="same")
        baseline = float(np.median(energy))
        mad = float(np.median(np.abs(energy - baseline))) + 1e-12
        threshold = baseline + self.threshold_mad * mad
        candidate_indices = np.flatnonzero(energy > threshold)
        peaks = []
        cursor = 0
        while cursor < len(candidate_indices):
            start = candidate_indices[cursor]
            stop = start + self.refractory
            group = candidate_indices[(candidate_indices >= start) &
                                      (candidate_indices < stop)]
            peak = int(group[np.argmax(energy[group])])
            strength = max(0.0, (float(energy[peak]) - baseline) / mad)
            confidence = min(1.0, strength / (self.threshold_mad * 2.0))
            peaks.append(AudioImpulse(
                start_time + peak / self.sample_rate, strength, confidence))
            cursor += len(group)
        return peaks


def analyze_audio_samples(samples: np.ndarray, sample_rate: int,
                          threshold_mad: float = 8.0) -> Dict:
    """Classify audio availability and return unclassified impact evidence."""
    samples = np.asarray(samples, dtype="<f4")
    if samples.size == 0 or float(np.max(np.abs(samples))) <= 1e-8:
        return {
            "status": "silent_audio", "sample_rate": sample_rate, "events": [],
            "warning": "Audio is silent; contacts must use visual evidence.",
        }
    events = AudioImpulseDetector(sample_rate, threshold_mad=threshold_mad).detect(samples)
    return {
        "status": "ok", "sample_rate": sample_rate,
        "events": [event.__dict__ for event in events],
        "warning": "Audio impulses are evidence only; they are not classified contacts.",
    }


def extract_video_audio_evidence(video_path: str, sample_rate: int = 8000) -> Dict:
    """Decode mono audio from a recording without failing visual analysis."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return {"status": "ffmpeg_unavailable", "sample_rate": sample_rate,
                "events": [], "warning": "Audio decoding is unavailable."}
    command = [
        ffmpeg, "-v", "error", "-i", video_path, "-vn", "-ac", "1",
        "-ar", str(sample_rate), "-f", "f32le", "-",
    ]
    try:
        completed = subprocess.run(command, check=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return {"status": "audio_decode_failed", "sample_rate": sample_rate,
                "events": [], "warning": "Audio decoding failed; using visual evidence."}
    if completed.returncode != 0:
        return {"status": "no_decodable_audio", "sample_rate": sample_rate,
                "events": [], "warning": "No decodable audio stream was found."}
    return analyze_audio_samples(np.frombuffer(completed.stdout, dtype="<f4"), sample_rate)
