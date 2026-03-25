"""Tests for LiveManager multi-camera constructor and single-camera backward compat."""
import pytest
from unittest.mock import MagicMock


def test_live_manager_multi_camera():
    from pipeline.live_manager import LiveManager
    from cv.camera_node import CameraNode

    cam1 = MagicMock(spec=CameraNode)
    cam1.camera_id = "cam1"
    cam1.quality_weight.return_value = 1.0

    cam2 = MagicMock(spec=CameraNode)
    cam2.camera_id = "cam2"
    cam2.quality_weight.return_value = 1.0

    manager = LiveManager(
        camera_nodes=[cam1, cam2],
        court_model=MagicMock(),
    )
    assert len(manager._camera_nodes) == 2
    assert manager._world_fusion is not None


def test_live_manager_single_camera_compat():
    from pipeline.live_manager import LiveManager

    analyzer = MagicMock()
    manager = LiveManager(analyzer=analyzer, device_id=0)
    assert manager._camera_nodes == []
    assert manager._analyzer == analyzer
