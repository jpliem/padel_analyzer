import numpy as np

from cv.audio_events import analyze_audio_samples


def test_runtime_audio_reports_silence_instead_of_fake_contacts():
    result = analyze_audio_samples(np.zeros(800), 8000)
    assert result["status"] == "silent_audio"
    assert result["events"] == []


def test_runtime_audio_returns_unclassified_impulse_evidence():
    samples = np.zeros(4000, dtype=np.float32)
    samples[2000:2005] = 1.0
    result = analyze_audio_samples(samples, 1000)
    assert result["status"] == "ok"
    assert len(result["events"]) == 1
    assert "not classified contacts" in result["warning"]
