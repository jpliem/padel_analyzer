import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from models.config import EventDetectorConfig



@pytest.fixture
def mock_deps():
    calibration = MagicMock()
    calibration.pixel_to_court.return_value = (5.0, 10.0)
    calibration.is_in_bounds.return_value = True
    calibration.is_in_service_box.return_value = True

    config = EventDetectorConfig()
    return calibration, config


class TestVideoAnalyzer:
    def test_process_frame_returns_frame_result(self, mock_deps):
        from pipeline.video_analyzer import VideoAnalyzer, FrameResult
        calibration, config = mock_deps

        with patch('pipeline.video_analyzer.UnifiedYoloDetector') as MockUnified:
            mock_unified = MockUnified.return_value
            mock_boxes = MagicMock(
                xyxy=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([]).reshape(0, 4))))),
                conf=MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([]))))),
            )
            mock_boxes.cls = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([])))))
            mock_unified.run.return_value = MagicMock(boxes=mock_boxes)

            va = VideoAnalyzer(
                match_id="test",
                calibration=calibration,
                config=config,
            )
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            result = va.process_frame(frame, frame_no=0)
            assert isinstance(result, FrameResult)
            assert result.frame_number == 0

    def test_auto_assignment_after_n_frames(self, mock_deps):
        from pipeline.video_analyzer import VideoAnalyzer
        calibration, config = mock_deps
        config.auto_assign_after_frames = 2

        with patch('pipeline.video_analyzer.UnifiedYoloDetector') as MockUnified:
            mock_unified = MockUnified.return_value
            mock_result = MagicMock()
            xyxy = np.array([[100,200,200,400],[300,200,400,400],[500,200,600,400],[700,200,800,400]])
            cls = np.array([0,0,0,0])
            conf = np.array([0.9,0.85,0.8,0.75])
            mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_result.boxes.cls.cpu.return_value.numpy.return_value = cls
            mock_result.boxes.conf.cpu.return_value.numpy.return_value = conf
            mock_unified.run.return_value = mock_result

            va = VideoAnalyzer(match_id="test", calibration=calibration, config=config)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)

            va.process_frame(frame, 0)
            assert not va._auto_assigned
            # Assignment happens at frame 30 (first multiple of 30 >= auto_assign_after_frames)
            for i in range(1, 30):
                va.process_frame(frame, i)
            assert not va._auto_assigned
            va.process_frame(frame, 30)
            assert va._auto_assigned

    def test_detector_type_tracknet(self, mock_deps):
        from pipeline.video_analyzer import VideoAnalyzer
        calibration, config = mock_deps

        with patch('pipeline.video_analyzer.UnifiedYoloDetector') as MockUnified, \
             patch('pipeline.video_analyzer.TrackNetBallDetector') as MockTrackNet:
            mock_unified = MockUnified.return_value
            mock_result = MagicMock()
            xyxy = np.array([]).reshape(0, 4)
            mock_result.boxes.xyxy.cpu.return_value.numpy.return_value = xyxy
            mock_result.boxes.cls.cpu.return_value.numpy.return_value = np.array([])
            mock_result.boxes.conf.cpu.return_value.numpy.return_value = np.array([])
            mock_unified.run.return_value = mock_result

            mock_tracknet = MockTrackNet.return_value
            mock_tracknet.detect.return_value = None

            va = VideoAnalyzer(
                match_id="test",
                calibration=calibration,
                config=config,
                detector_type="tracknet",
            )
            # Should have created TrackNetBallDetector
            MockTrackNet.assert_called_once()
