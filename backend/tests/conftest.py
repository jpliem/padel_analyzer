import pytest
import numpy as np
from models.types import MatchConfig, MatchFormat, ServerInfo, TeamId


@pytest.fixture
def default_match_config():
    return MatchConfig()


@pytest.fixture
def golden_point_config():
    return MatchConfig(golden_point=True)


@pytest.fixture
def advantage_config():
    return MatchConfig(golden_point=False)


@pytest.fixture
def best_of_1_config():
    return MatchConfig(format=MatchFormat.BEST_OF_1)


@pytest.fixture
def sample_court_corners_pixels():
    """4 court corners as they might appear in a 1080p frame (behind-baseline camera).
    Near baseline (y=0 in court) = bottom of frame (large pixel y).
    Far baseline (y=20 in court) = top of frame (small pixel y)."""
    return np.array([
        [320, 700],   # near-left (bottom-left of frame)
        [1600, 700],  # near-right (bottom-right of frame)
        [1200, 200],  # far-right (top-right, narrower due to perspective)
        [720, 200],   # far-left (top-left, narrower due to perspective)
    ], dtype=np.float32)


@pytest.fixture
def court_real_coords():
    """Real-world court corners in meters (10x20m padel court)."""
    return np.array([
        [0, 0],
        [10, 0],
        [10, 20],
        [0, 20],
    ], dtype=np.float32)


@pytest.fixture
def court_model():
    from models.court_model import PadelCourtModel
    return PadelCourtModel()


@pytest.fixture
def calibrated_camera_model():
    from cv.camera_model import CameraModel
    cam = CameraModel()
    keypoints = [
        [100, 600], [900, 600],
        [200, 450], [500, 450], [800, 450],
        [250, 350], [750, 350],
        [200, 250], [500, 250], [800, 250],
        [100, 100], [900, 100],
    ]
    cam.calibrate(keypoints, image_width=1000, image_height=700)
    return cam
