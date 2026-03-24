import pytest


class TestGetDevice:
    def test_returns_string(self):
        from cv.detectors.device import get_device
        device = get_device()
        assert isinstance(device, str)
        assert device in ("mps", "cuda", "cpu")

    def test_cpu_fallback(self, monkeypatch):
        import torch
        from cv.detectors.device import get_device
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        device = get_device()
        assert device == "cpu"
