import pytest
import torch
import numpy as np
from unittest.mock import MagicMock


class TestTrackNetV2Model:
    def test_forward_pass_shape(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model(in_dim=27, out_dim=8)
        model.eval()
        x = torch.randn(1, 27, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 8, 288, 512)

    def test_output_range_0_to_1(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model(in_dim=27, out_dim=8)
        model.eval()
        x = torch.randn(1, 27, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_support(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model(in_dim=27, out_dim=8)
        model.eval()
        x = torch.randn(2, 27, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 8, 288, 512)


class TestTrackNetBallDetector:
    def _make_detector_with_mock_model(self, peak_conf=0.9, peak_x=256, peak_y=144):
        from cv.detectors.tracknet import TrackNetBallDetector, TrackNetV2Model

        mock_model = MagicMock(spec=TrackNetV2Model)
        mock_model.out_channels = 8
        # Output: (1, 8, 288, 512) — last channel is the current frame heatmap
        output = torch.zeros(1, 8, 288, 512)
        # Plant a cluster so Gaussian blur preserves it
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                py = max(0, min(287, peak_y + dy))
                px = max(0, min(511, peak_x + dx))
                output[0, -1, py, px] = peak_conf
        mock_model.return_value = output
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        detector = TrackNetBallDetector.__new__(TrackNetBallDetector)
        detector._model = mock_model
        detector._device_str = "cpu"
        detector._torch_device = torch.device("cpu")
        detector._buffer = []
        detector._conf_threshold = 0.5
        detector._yolo_fallback = None
        return detector

    def test_returns_none_for_early_frames(self):
        detector = self._make_detector_with_mock_model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(8):  # needs 9 frames
            assert detector.detect(frame, i) is None

    def test_returns_bbox_after_enough_frames(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.9, peak_x=256, peak_y=144)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(8):
            detector.detect(frame, i)
        result = detector.detect(frame, 8)
        assert result is not None
        assert len(result) == 4

    def test_returns_none_when_low_confidence(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.1)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(9):
            result = detector.detect(frame, i)
        assert result is None

    def test_falls_back_to_yolo_early_frames(self):
        detector = self._make_detector_with_mock_model()
        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = [100, 200, 120, 220]
        detector._yolo_fallback = mock_yolo
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(frame, 0)
        assert result == [100, 200, 120, 220]

    def test_buffer_stays_at_9(self):
        detector = self._make_detector_with_mock_model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(15):
            detector.detect(frame, i)
        assert len(detector._buffer) == 9

    def test_warm_up_does_not_affect_buffer(self):
        detector = self._make_detector_with_mock_model()
        detector.warm_up()
        assert len(detector._buffer) == 0

    def test_device_property(self):
        detector = self._make_detector_with_mock_model()
        assert detector.device == "cpu"
