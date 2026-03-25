import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch


class TestTrackNetV2Model:
    def test_forward_pass_shape(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        # Input: batch=1, 9 channels (3 frames × 3 RGB), 360×640
        x = torch.randn(1, 9, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 288, 512)  # 3 heatmaps (one per input frame)

    def test_output_range_0_to_1(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        x = torch.randn(1, 9, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_batch_support(self):
        from cv.detectors.tracknet import TrackNetV2Model
        model = TrackNetV2Model()
        model.eval()
        x = torch.randn(2, 9, 288, 512)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3, 288, 512)


class TestTrackNetBallDetector:
    def _make_detector_with_mock_model(self, peak_conf=0.9, peak_x=320, peak_y=180):
        """Create detector with a mock model that returns a heatmap with a planted peak."""
        from cv.detectors.tracknet import TrackNetBallDetector, TrackNetV2Model

        # Create a mock model that returns a heatmap with a peak at the specified location
        mock_model = MagicMock(spec=TrackNetV2Model)
        heatmap = torch.zeros(1, 3, 288, 512)
        heatmap[0, 2, peak_y, peak_x] = peak_conf  # channel 2 = most recent frame
        mock_model.return_value = heatmap
        mock_model.eval = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)

        detector = TrackNetBallDetector.__new__(TrackNetBallDetector)
        detector._model = mock_model
        detector._device_str = "cpu"
        detector._torch_device = torch.device("cpu")
        detector._buffer = []
        detector._conf_threshold = 0.05
        detector._yolo_fallback = None
        return detector

    def test_returns_none_for_first_two_frames(self):
        detector = self._make_detector_with_mock_model()
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        assert detector.detect(frame, 0) is None
        assert detector.detect(frame, 1) is None

    def test_returns_bbox_on_third_frame(self):
        detector = self._make_detector_with_mock_model(peak_conf=0.9, peak_x=256, peak_y=144)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        detector.detect(frame, 0)
        detector.detect(frame, 1)
        result = detector.detect(frame, 2)
        assert result is not None
        assert len(result) == 4
        # Peak at heatmap (256, 144) scaled to frame (1280, 720):
        # scale_x = 1280/512 = 2.5, scale_y = 720/288 = 2.5
        # cx = 256*2.5 = 640, cy = 144*2.5 = 360
        expected_cx, expected_cy = 640, 360
        assert abs(result[0] - (expected_cx - 10)) < 2
        assert abs(result[1] - (expected_cy - 10)) < 2

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
