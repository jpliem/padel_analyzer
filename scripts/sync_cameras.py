#!/usr/bin/env python
"""Time-sync two camera videos via audio cross-correlation.

The PADELVIC cameras film the same match but start at different moments and run
at different frame rates. To triangulate, we need to know which frame of camera
B corresponds to a frame of camera A. Their audio tracks share the same sound
(ball hits, voices), so cross-correlating the waveforms recovers the time offset
robustly — no timecode or manual event-picking needed.

Outputs the offset in seconds and a per-camera frame mapping, written to JSON.

Example:
    python scripts/sync_cameras.py \
        --ref  data/datasets/padelvic/cameras/panasonic_final.mp4 \
        --other data/datasets/padelvic/cameras/gopro.mp4 \
        --out /tmp/sync_pana_gopro.json
"""
import sys
import os
import argparse
import json
import subprocess
import tempfile
import wave

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE_RATE = 8000  # mono, plenty for ball-hit transients


def extract_audio(video: str, duration_s: float, start_s: float = 0.0) -> np.ndarray:
    """Extract mono PCM audio via ffmpeg → float32 array at SAMPLE_RATE."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    try:
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", str(start_s), "-t", str(duration_s),
               "-i", video, "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-acodec", "pcm_s16le", wav_path]
        subprocess.run(cmd, check=True)
        with wave.open(wav_path, "rb") as w:
            frames = w.readframes(w.getnframes())
        a = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        a -= a.mean()
        n = np.linalg.norm(a)
        return a / n if n > 0 else a
    finally:
        os.unlink(wav_path)


def fps_of(video: str) -> float:
    import cv2
    c = cv2.VideoCapture(video)
    fps = c.get(cv2.CAP_PROP_FPS)
    c.release()
    return fps or 30.0


def motion_series(video: str, window_s: float, rate_hz: float = 2.0,
                  start_s: float = 0.0, size=(160, 90)) -> np.ndarray:
    """Per-time-step motion energy = mean abs frame-difference of small grayscale.

    Sampled at a fixed TIME rate (independent of each camera's fps) so the two
    series share a time base. Rallies spike, dead time is flat — a temporal
    fingerprint shared across cameras. Visual alternative to audio sync.
    """
    import cv2
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = fps / rate_hz
    n = int(window_s * rate_hz)
    prev = None
    series = []
    for i in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_s * fps + i * step))
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size).astype(np.float32)
        if prev is not None:
            series.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    a = np.asarray(series, dtype=np.float32)
    if len(a):
        a -= a.mean()
        nrm = np.linalg.norm(a)
        if nrm > 0:
            a /= nrm
    return a


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio-sync two camera videos.")
    ap.add_argument("--ref", required=True, help="Reference video")
    ap.add_argument("--other", required=True, help="Video to align to ref")
    ap.add_argument("--window", type=float, default=300.0,
                    help="Seconds to correlate (default 300)")
    ap.add_argument("--start", type=float, default=0.0, help="Start offset seconds")
    ap.add_argument("--method", choices=["audio", "motion"], default="audio",
                    help="audio cross-corr (needs sound) or visual motion-energy")
    ap.add_argument("--rate", type=float, default=2.0,
                    help="motion sampling rate Hz (default 2)")
    ap.add_argument("--out", help="Write sync JSON here")
    args = ap.parse_args()

    ref = args.ref if os.path.isabs(args.ref) else os.path.join(_ROOT, args.ref)
    other = args.other if os.path.isabs(args.other) else os.path.join(_ROOT, args.other)
    for p in (ref, other):
        if not os.path.exists(p):
            print(f"ERROR: not found: {p}", file=sys.stderr)
            return 1

    if args.method == "audio":
        print(f"extracting {args.window}s audio from both…", file=sys.stderr)
        a = extract_audio(ref, args.window, args.start)
        b = extract_audio(other, args.window, args.start)
        rate = SAMPLE_RATE
    else:
        print(f"computing {args.window}s motion energy from both @ {args.rate}Hz…",
              file=sys.stderr)
        a = motion_series(ref, args.window, args.rate, args.start)
        b = motion_series(other, args.window, args.rate, args.start)
        rate = args.rate

    from scipy.signal import fftconvolve
    # cross-correlation via FFT: corr[k] peak => lag of b relative to a
    corr = fftconvolve(a, b[::-1], mode="full")
    lag = int(np.argmax(corr) - (len(b) - 1))
    offset_s = lag / rate  # positive => 'other' is delayed vs ref by this many seconds
    # normalized peak (both inputs are unit-norm) => correlation coefficient
    peak = float(np.max(corr))

    fps_ref, fps_other = fps_of(ref), fps_of(other)

    result = {
        "ref": os.path.basename(ref),
        "other": os.path.basename(other),
        "offset_seconds": round(offset_s, 4),
        "correlation_peak": round(peak, 4),
        "fps_ref": fps_ref,
        "fps_other": fps_other,
        "note": "match time t (s) -> ref frame = t*fps_ref ; "
                "other frame = (t - offset_seconds)*fps_other",
    }
    print("\n=== camera sync ===")
    print(f"  {result['ref']}  fps={fps_ref}")
    print(f"  {result['other']}  fps={fps_other}")
    print(f"  audio offset: {offset_s:+.3f} s  (peak corr {peak:.3f})")
    print(f"  -> '{result['other']}' is {'delayed' if offset_s>0 else 'ahead'} "
          f"by {abs(offset_s):.3f}s vs '{result['ref']}'")

    if args.out:
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
