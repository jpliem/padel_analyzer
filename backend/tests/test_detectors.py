import pytest
from unittest.mock import MagicMock, patch
import numpy as np


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


class TestUnifiedYoloDetector:
    def test_caches_results_for_same_frame(self):
        from cv.detectors.yolo import UnifiedYoloDetector
        detector = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        mock_model = MagicMock()
        mock_results = [MagicMock()]
        mock_model.return_value = mock_results
        detector.model = mock_model
        detector._cache = (None, None)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        r1 = detector.run(frame, frame_id=0)
        r2 = detector.run(frame, frame_id=0)
        assert mock_model.call_count == 1

    def test_new_frame_runs_inference(self):
        from cv.detectors.yolo import UnifiedYoloDetector
        detector = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        mock_model = MagicMock()
        mock_model.return_value = [MagicMock()]
        detector.model = mock_model
        detector._cache = (None, None)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.run(frame, frame_id=0)
        detector.run(frame, frame_id=1)
        assert mock_model.call_count == 2


class TestYoloBallDetector:
    def _make_mock_result(self, boxes_data):
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        if len(boxes_data) == 0:
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([])
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([])
        else:
            xyxy = np.array([b[:4] for b in boxes_data])
            cls = np.array([b[4] for b in boxes_data])
            conf = np.array([b[5] for b in boxes_data])
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = conf
        mock_result.boxes = mock_boxes
        return mock_result

    def test_detects_ball_cls32(self):
        from cv.detectors.yolo import YoloBallDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([[100, 200, 120, 220, 32, 0.8]])
        unified.model.return_value = [result]

        detector = YoloBallDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame, frame_id=0)
        assert bbox is not None
        assert len(bbox) == 4

    def test_returns_none_when_no_ball(self):
        from cv.detectors.yolo import YoloBallDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([[100, 200, 300, 400, 0, 0.9]])
        unified.model.return_value = [result]

        detector = YoloBallDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame, frame_id=0)
        assert bbox is None

    def test_filters_low_confidence(self):
        from cv.detectors.yolo import YoloBallDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([[100, 200, 120, 220, 32, 0.1]])
        unified.model.return_value = [result]

        detector = YoloBallDetector(unified, conf_threshold=0.3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = detector.detect(frame, frame_id=0)
        assert bbox is None


class TestYoloPlayerDetector:
    def _make_mock_result(self, boxes_data):
        mock_result = MagicMock()
        mock_boxes = MagicMock()
        if len(boxes_data) == 0:
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([]).reshape(0, 4)
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([])
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([])
        else:
            xyxy = np.array([b[:4] for b in boxes_data])
            cls = np.array([b[4] for b in boxes_data])
            conf = np.array([b[5] for b in boxes_data])
            mock_boxes.xyxy = MagicMock()
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_boxes.cls = MagicMock()
            mock_boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_boxes.conf = MagicMock()
            mock_boxes.conf.cpu.return_value.numpy.return_value = conf
        mock_result.boxes = mock_boxes
        return mock_result

    def test_detects_persons_cls0(self):
        from cv.detectors.yolo import YoloPlayerDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([
            [100, 200, 200, 400, 0, 0.9],
            [300, 200, 400, 400, 0, 0.85],
        ])
        unified.model.return_value = [result]

        detector = YoloPlayerDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_id=0)
        assert detections.shape[0] == 2
        assert detections.shape[1] == 6

    def test_max_4_detections(self):
        from cv.detectors.yolo import YoloPlayerDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([
            [100, 200, 200, 400, 0, 0.9],
            [300, 200, 400, 400, 0, 0.85],
            [500, 200, 600, 400, 0, 0.8],
            [700, 200, 800, 400, 0, 0.75],
            [900, 200, 1000, 400, 0, 0.6],
        ])
        unified.model.return_value = [result]

        detector = YoloPlayerDetector(unified, max_detections=4)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_id=0)
        assert detections.shape[0] == 4

    def test_returns_empty_when_no_persons(self):
        from cv.detectors.yolo import YoloPlayerDetector, UnifiedYoloDetector
        unified = UnifiedYoloDetector.__new__(UnifiedYoloDetector)
        unified.model = MagicMock()
        unified._cache = (None, None)
        result = self._make_mock_result([[100, 200, 120, 220, 32, 0.8]])
        unified.model.return_value = [result]

        detector = YoloPlayerDetector(unified)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame, frame_id=0)
        assert detections.shape[0] == 0
