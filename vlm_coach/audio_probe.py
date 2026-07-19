from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import find_peaks, stft


def spectral_impulses(samples: np.ndarray, sample_rate: int,
                      minimum_gap: float = 0.18) -> dict:
    """Find sharp broadband onsets; candidates are not yet padel-hit labels."""
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32)
    scale = float(np.max(np.abs(samples))) if samples.size else 0.0
    if scale:
        samples /= scale
    _, times, spectrum = stft(
        samples, fs=sample_rate, nperseg=2048, noverlap=1536,
        boundary=None, padded=False,
    )
    frequencies = np.fft.rfftfreq(2048, 1 / sample_rate)
    band = (frequencies >= 1500) & (frequencies <= min(12000, sample_rate / 2))
    magnitude = np.log1p(30.0 * np.abs(spectrum[band]))
    flux = np.maximum(0.0, np.diff(magnitude, axis=1)).mean(axis=0)
    if not flux.size:
        return {"times": [], "strengths": [], "threshold": None}
    median = float(np.median(flux))
    mad = float(np.median(np.abs(flux - median))) or 1e-6
    threshold = median + 5.0 * mad
    hop_seconds = 512 / sample_rate
    peaks, properties = find_peaks(
        flux, height=threshold, prominence=3.0 * mad,
        distance=max(1, int(minimum_gap / hop_seconds)),
    )
    peak_times = [round(float(times[index + 1]), 3) for index in peaks]
    strengths = [round(float(value), 5) for value in properties["peak_heights"]]
    return {"times": peak_times, "strengths": strengths,
            "threshold": round(threshold, 5)}


def cadence_summary(times: list[float], duration: float) -> dict:
    intervals = [later - earlier for earlier, later in zip(times, times[1:])]
    plausible = [value for value in intervals if 0.25 <= value <= 3.0]
    return {
        "candidate_impulses": len(times),
        "candidates_per_second": round(len(times) / duration, 3) if duration else 0.0,
        "plausible_rally_intervals": len(plausible),
        "median_interval_seconds": round(float(np.median(plausible)), 3)
        if plausible else None,
        "warning": "Untrained spectral impulses include speech, footsteps, glass and music.",
    }


def quiet_intervals(times: list[float], duration: float,
                    minimum: float = 2.0) -> list[list[float]]:
    """Return long internal gaps between hit-like impulses."""
    result = []
    for earlier, later in zip(times, times[1:]):
        if later - earlier >= minimum:
            result.append([round(earlier, 3), round(later, 3)])
    return result


def read_audio_clip(video: Path, start: float, duration: float) -> tuple[int, np.ndarray]:
    with tempfile.TemporaryDirectory(prefix="padel-audio-") as temporary:
        wav = Path(temporary) / "clip.wav"
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(start), "-i", str(video), "-t", str(duration),
            "-vn", "-ac", "1", "-ar", "44100", "-c:a", "pcm_s16le", str(wav),
        ], check=True)
        return wavfile.read(wav)


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe padel audio for hit-like impulses")
    parser.add_argument("--video", required=True)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    video = Path(args.video).resolve()
    sample_rate, samples = read_audio_clip(video, args.start, args.duration)
    impulses = spectral_impulses(samples, sample_rate)
    output = {
        "video": str(video), "start": args.start, "duration": args.duration,
        "candidate_times_relative": impulses["times"],
        "candidate_times_absolute": [round(args.start + value, 3) for value in impulses["times"]],
        "strengths": impulses["strengths"], "threshold": impulses["threshold"],
        "cadence": cadence_summary(impulses["times"], args.duration),
        "quiet_intervals_relative": quiet_intervals(impulses["times"], args.duration),
    }
    rendered = json.dumps(output, indent=2)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
