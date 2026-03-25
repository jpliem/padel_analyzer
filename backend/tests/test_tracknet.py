import pytest
import torch
import numpy as np


class TestTrackNetV2Model:
    def test_forward_pass_shape(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        # Input: batch=1, 9 channels (3 frames × 3 RGB), 360×640
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1, 360, 640)

    def test_output_range_0_to_1(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_support(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        x = torch.randn(2, 9, 360, 640)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 360, 640)
