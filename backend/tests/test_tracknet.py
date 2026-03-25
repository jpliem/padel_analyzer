import pytest
import torch
import numpy as np
from unittest.mock import MagicMock


class TestTrackNetV2Model:
    def test_forward_pass_shape(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model(out_channels=256)
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x, testing=True)
        assert out.shape == (1, 256, 230400)  # 360*640 = 230400

    def test_output_range_0_to_1(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model(out_channels=256)
        model.eval()
        x = torch.randn(1, 9, 360, 640)
        with torch.no_grad():
            out = model(x, testing=True)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_support(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model(out_channels=256)
        model.eval()
        x = torch.randn(2, 9, 360, 640)
        with torch.no_grad():
            out = model(x, testing=True)
        assert out.shape == (2, 256, 230400)


class TestTrackNetBallDetector:
    def _make_detector_with_mock_model(self, peak_conf=0.9, peak_x=320, peak_y=180):
        from cv.detectors.tracknet import TrackNetBallDetector, TrackNetV2Model

        mock_model = MagicMock(spec=TrackNetV2Model)
        mock_model.out_channels = 256
        # Create output: (1, 256, 230400) where argmax at (peak_y, peak_x) gives high class
        # Plant a cluster of high-class pixels so Gaussian blur preserves the peak
        output = torch.zeros(1, 256, 360 * 640)
        high_class = int(peak_conf * 255)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                py = max(0, min(359, peak_y + dy))
                px = max(0, min(639, peak_x + dx))
                pixel_idx = py * 640 + px
                output[0, high_class, pixel_idx] = 1.0
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

    def test_returns_none_for_first_two_frames(self):
        detector = self._make_detector_with_mock_model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect(frame, 0) is None
        assert detector.detect(frame, 1) is None

    def test_returns_bbox_on_third_frame(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.9, peak_x=320, peak_y=180)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0)
        detector.detect(frame, 1)
        result = detector.detect(frame, 2)
        assert result is not None
        assert len(result) == 4

    def test_returns_none_when_low_confidence(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.1)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0)
        detector.detect(frame, 1)
        result = detector.detect(frame, 2)
        assert result is None

    def test_falls_back_to_yolo_first_frames(self):
        detector = self._make_detector_with_mock_model()
        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = [100, 200, 120, 220]
        detector._yolo_fallback = mock_yolo
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(frame, 0)
        assert result == [100, 200, 120, 220]
        mock_yolo.detect.assert_called_once()

    def test_falls_back_to_yolo_low_confidence(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.1)
        mock_yolo = MagicMock()
        mock_yolo.detect.return_value = [100, 200, 120, 220]
        detector._yolo_fallback = mock_yolo
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0)
        detector.detect(frame, 1)
        result = detector.detect(frame, 2)
        assert result == [100, 200, 120, 220]

    def test_buffer_stays_at_3(self):
        detector = self._make_detector_with_mock_model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        for i in range(10):
            detector.detect(frame, i)
        assert len(detector._buffer) == 3

    def test_warm_up_does_not_affect_buffer(self):
        detector = self._make_detector_with_mock_model()
        detector.warm_up()
        assert len(detector._buffer) == 0

    def test_device_property(self):
        detector = self._make_detector_with_mock_model()
        assert detector.device == "cpu"
