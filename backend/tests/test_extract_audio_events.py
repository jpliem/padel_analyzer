import numpy as np

from scripts.extract_audio_events import analyze_samples


def test_silent_audio_is_reported_explicitly():
    payload = analyze_samples(np.zeros(160), "match.mp4", 16000, 8.0)

    assert payload["status"] == "silent_audio"
    assert payload["events"] == []
    assert "visual evidence" in payload["warning"]


def test_non_silent_audio_is_processed():
    samples = np.zeros(4000, dtype=np.float32)
    samples[2000:2005] = 1.0

    payload = analyze_samples(samples, "match.mp4", 1000, 8.0)

    assert payload["status"] == "ok"
    assert len(payload["events"]) == 1
