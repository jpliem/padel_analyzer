"""Tests for CameraModel adapter methods (CourtCalibration interface)."""

import pytest
import numpy as np
from unittest.mock import patch

from cv.camera_model import CameraModel, COURT_WIDTH, COURT_LENGTH, NET_Y


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def homography_camera():
    """CameraModel calibrated with 4 ground keypoints — uses homography fallback.

    Pixel layout (1280x720 frame, behind-baseline camera):
      near baseline (court Y=0)  → bottom of frame (pixel y ≈ 680)
      far  baseline (court Y=20) → top of frame    (pixel y ≈ 40)
      left  (court X=0)          → left of frame   (pixel x ≈ 100)
      right (court X=10)         → right of frame  (pixel x ≈ 1180)
    """
    cam = CameraModel()
    # 12 ground keypoints in pixel space (approximate perspective layout)
    keypoints_2d = [
        [100, 680],   # k1  near-left  baseline  (0,   0)
        [1180, 680],  # k2  near-right baseline  (10,  0)
        [280, 500],   # k3  near-left  service   (0,   6.95)
        [640, 500],   # k4  near-center service  (5,   6.95)
        [1000, 500],  # k5  near-right service   (10,  6.95)
        [240, 400],   # k6  net-left  ground     (0,  10)
        [1040, 400],  # k7  net-right ground     (10, 10)
        [300, 310],   # k8  far-left  service    (0,  13.05)
        [640, 310],   # k9  far-center service   (5,  13.05)
        [980, 310],   # k10 far-right service    (10, 13.05)
        [380, 160],   # k11 far-left  baseline   (0,  20)
        [900, 160],   # k12 far-right baseline   (10, 20)
    ]
    cam.calibrate(keypoints_2d, net_top_2d=None, image_width=1280, image_height=720)
    assert cam.homography is not None, "Homography should be set with 12 ground keypoints"
    return cam


@pytest.fixture
def calibrated_3d_camera():
    """CameraModel calibrated with 12 ground + 2 net-top keypoints — full 3D model."""
    cam = CameraModel()
    keypoints_2d = [
        [100, 680],
        [1180, 680],
        [280, 500],
        [640, 500],
        [1000, 500],
        [240, 400],
        [1040, 400],
        [300, 310],
        [640, 310],
        [980, 310],
        [380, 160],
        [900, 160],
    ]
    net_top_2d = [
        [230, 370],   # net-left top  (0,  10, 0.92)
        [1050, 370],  # net-right top (10, 10, 0.92)
    ]
    cam.calibrate(keypoints_2d, net_top_2d=net_top_2d, image_width=1280, image_height=720)
    return cam


# ---------------------------------------------------------------------------
# pixel_to_court — delegates to project_to_ground
# ---------------------------------------------------------------------------

class TestPixelToCourt:
    def test_delegates_to_project_to_ground(self, homography_camera):
        """pixel_to_court should return the same result as project_to_ground."""
        px, py = 640.0, 400.0
        result_adapter = homography_camera.pixel_to_court(px, py)
        result_direct = homography_camera.project_to_ground(px, py)
        assert result_adapter == pytest.approx(result_direct, abs=1e-9)

    def test_center_pixel_gives_near_center_court(self, homography_camera):
        """The net-center pixel should map somewhere near court center (5, 10)."""
        # Net ground center is approximately pixel (640, 400) in our layout
        cx, cy = homography_camera.pixel_to_court(640.0, 400.0)
        assert 0.0 <= cx <= COURT_WIDTH, f"cx={cx} out of court bounds"
        assert 0.0 <= cy <= COURT_LENGTH, f"cy={cy} out of court bounds"
        # Should be roughly near the middle of the court
        assert abs(cx - 5.0) < 3.0, f"cx={cx} too far from center (5.0)"
        assert abs(cy - 10.0) < 5.0, f"cy={cy} too far from net (10.0)"

    def test_near_baseline_pixel_maps_to_low_court_y(self, homography_camera):
        """Bottom-of-frame pixels (near baseline) should map to low court Y values."""
        _, cy = homography_camera.pixel_to_court(640.0, 680.0)
        assert cy < 5.0, f"Near-baseline pixel should map to cy < 5, got {cy}"

    def test_far_baseline_pixel_maps_to_high_court_y(self, homography_camera):
        """Top-of-frame pixels (far baseline) should map to high court Y values."""
        _, cy = homography_camera.pixel_to_court(640.0, 160.0)
        assert cy > 15.0, f"Far-baseline pixel should map to cy > 15, got {cy}"

    def test_returns_tuple_of_two_floats(self, homography_camera):
        result = homography_camera.pixel_to_court(640.0, 360.0)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# court_to_pixel_2d — delegates to court_to_pixel(..., 0)
# ---------------------------------------------------------------------------

class TestCourtToPixel2D:
    def test_delegates_to_court_to_pixel_at_z0(self, homography_camera):
        """court_to_pixel_2d(cx, cy) should equal court_to_pixel(cx, cy, 0.0)."""
        cx, cy = 5.0, 10.0
        result_adapter = homography_camera.court_to_pixel_2d(cx, cy)
        result_direct = homography_camera.court_to_pixel(cx, cy, 0.0)
        assert result_adapter == pytest.approx(result_direct, abs=1e-9)

    def test_court_to_pixel_2d_uses_mock(self, homography_camera):
        """Verify court_to_pixel_2d actually calls court_to_pixel with cz=0."""
        with patch.object(homography_camera, "court_to_pixel",
                          wraps=homography_camera.court_to_pixel) as mock_ctp:
            homography_camera.court_to_pixel_2d(5.0, 10.0)
            mock_ctp.assert_called_once_with(5.0, 10.0, 0.0)

    def test_returns_tuple_of_two_floats(self, homography_camera):
        result = homography_camera.court_to_pixel_2d(5.0, 10.0)
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_roundtrip_near_baseline(self, homography_camera):
        """court_to_pixel_2d then pixel_to_court should recover original coords."""
        court_x, court_y = 2.0, 3.0
        px, py = homography_camera.court_to_pixel_2d(court_x, court_y)
        rx, ry = homography_camera.pixel_to_court(px, py)
        assert rx == pytest.approx(court_x, abs=0.1)
        assert ry == pytest.approx(court_y, abs=0.1)

    def test_roundtrip_far_side(self, homography_camera):
        """Roundtrip should work for far side of court."""
        court_x, court_y = 8.0, 17.0
        px, py = homography_camera.court_to_pixel_2d(court_x, court_y)
        rx, ry = homography_camera.pixel_to_court(px, py)
        assert rx == pytest.approx(court_x, abs=0.1)
        assert ry == pytest.approx(court_y, abs=0.1)


# ---------------------------------------------------------------------------
# is_in_bounds
# ---------------------------------------------------------------------------

class TestIsInBounds:
    def test_center_court_is_in_bounds(self, homography_camera):
        assert homography_camera.is_in_bounds(5.0, 10.0) is True

    def test_near_left_corner_is_in_bounds(self, homography_camera):
        assert homography_camera.is_in_bounds(0.0, 0.0) is True

    def test_far_right_corner_is_in_bounds(self, homography_camera):
        assert homography_camera.is_in_bounds(10.0, 20.0) is True

    def test_negative_x_out_of_bounds(self, homography_camera):
        assert homography_camera.is_in_bounds(-0.1, 10.0) is False

    def test_negative_y_out_of_bounds(self, homography_camera):
        assert homography_camera.is_in_bounds(5.0, -0.1) is False

    def test_x_beyond_width_out_of_bounds(self, homography_camera):
        assert homography_camera.is_in_bounds(10.1, 10.0) is False

    def test_y_beyond_length_out_of_bounds(self, homography_camera):
        assert homography_camera.is_in_bounds(5.0, 20.1) is False

    def test_boundary_values_in_bounds(self, homography_camera):
        # Exact boundary should be in-bounds (inclusive)
        assert homography_camera.is_in_bounds(0.0, 0.0) is True
        assert homography_camera.is_in_bounds(COURT_WIDTH, COURT_LENGTH) is True

    def test_uses_court_constants(self, homography_camera):
        assert homography_camera.is_in_bounds(COURT_WIDTH - 0.01, COURT_LENGTH - 0.01) is True
        assert homography_camera.is_in_bounds(COURT_WIDTH + 0.01, COURT_LENGTH) is False


# ---------------------------------------------------------------------------
# get_court_side
# ---------------------------------------------------------------------------

class TestGetCourtSide:
    def test_near_side_y_below_net(self, homography_camera):
        assert homography_camera.get_court_side(5.0, 0.0) == "near"

    def test_near_side_just_below_net(self, homography_camera):
        assert homography_camera.get_court_side(5.0, NET_Y - 0.001) == "near"

    def test_far_side_at_net(self, homography_camera):
        # y == NET_Y should be "far" (not < NET_Y)
        assert homography_camera.get_court_side(5.0, NET_Y) == "far"

    def test_far_side_beyond_net(self, homography_camera):
        assert homography_camera.get_court_side(5.0, 15.0) == "far"

    def test_far_side_far_baseline(self, homography_camera):
        assert homography_camera.get_court_side(5.0, 20.0) == "far"

    def test_near_side_various_x(self, homography_camera):
        for x in [0.0, 2.5, 5.0, 7.5, 10.0]:
            assert homography_camera.get_court_side(x, 5.0) == "near"

    def test_far_side_various_x(self, homography_camera):
        for x in [0.0, 2.5, 5.0, 7.5, 10.0]:
            assert homography_camera.get_court_side(x, 15.0) == "far"

    def test_returns_string(self, homography_camera):
        result = homography_camera.get_court_side(5.0, 5.0)
        assert isinstance(result, str)
        assert result in ("near", "far")


# ---------------------------------------------------------------------------
# is_in_service_box
# ---------------------------------------------------------------------------

class TestIsInServiceBox:
    def test_near_left_box_center(self, homography_camera):
        # near_left: x in [0,5], y in [6.95, 10]
        assert homography_camera.is_in_service_box(2.5, 8.0, "near_left") is True

    def test_near_right_box_center(self, homography_camera):
        # near_right: x in [5,10], y in [6.95, 10]
        assert homography_camera.is_in_service_box(7.5, 8.0, "near_right") is True

    def test_far_left_box_center(self, homography_camera):
        # far_left: x in [0,5], y in [10, 13.05]
        assert homography_camera.is_in_service_box(2.5, 11.5, "far_left") is True

    def test_far_right_box_center(self, homography_camera):
        # far_right: x in [5,10], y in [10, 13.05]
        assert homography_camera.is_in_service_box(7.5, 11.5, "far_right") is True

    def test_point_in_wrong_box(self, homography_camera):
        # near_left center should not be in near_right
        assert homography_camera.is_in_service_box(2.5, 8.0, "near_right") is False

    def test_point_outside_all_boxes(self, homography_camera):
        # Center of court baseline — not in any service box
        assert homography_camera.is_in_service_box(5.0, 1.0, "near_left") is False
        assert homography_camera.is_in_service_box(5.0, 1.0, "near_right") is False

    def test_unknown_box_returns_false(self, homography_camera):
        assert homography_camera.is_in_service_box(5.0, 8.0, "invalid_box") is False
        assert homography_camera.is_in_service_box(5.0, 8.0, "") is False

    def test_boundary_inclusive(self, homography_camera):
        # Exact boundary of near_left box (x=0, y=6.95)
        assert homography_camera.is_in_service_box(0.0, 6.95, "near_left") is True
        assert homography_camera.is_in_service_box(5.0, 10.0, "near_left") is True

    def test_just_outside_service_box_y(self, homography_camera):
        # y just below 6.95 should be outside near service boxes
        assert homography_camera.is_in_service_box(2.5, 6.94, "near_left") is False

    def test_all_valid_box_names(self, homography_camera):
        valid_boxes = ["near_left", "near_right", "far_left", "far_right"]
        for box in valid_boxes:
            # Each should return a bool without raising
            result = homography_camera.is_in_service_box(5.0, 8.5, box)
            assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Adapter methods work with 3D camera model too
# ---------------------------------------------------------------------------

class TestAdaptersWith3DCamera:
    def test_pixel_to_court_3d_delegates_to_project_to_ground(self, calibrated_3d_camera):
        if not calibrated_3d_camera.has_3d():
            pytest.skip("3D calibration not available with this fixture")
        px, py = 640.0, 400.0
        result_adapter = calibrated_3d_camera.pixel_to_court(px, py)
        result_direct = calibrated_3d_camera.project_to_ground(px, py)
        assert result_adapter == pytest.approx(result_direct, abs=1e-9)

    def test_is_in_bounds_does_not_require_calibration(self):
        """is_in_bounds is purely geometric, doesn't need a calibrated camera."""
        cam = CameraModel()  # uncalibrated
        assert cam.is_in_bounds(5.0, 10.0) is True
        assert cam.is_in_bounds(-1.0, 10.0) is False

    def test_get_court_side_does_not_require_calibration(self):
        """get_court_side is purely geometric, doesn't need a calibrated camera."""
        cam = CameraModel()  # uncalibrated
        assert cam.get_court_side(5.0, 5.0) == "near"
        assert cam.get_court_side(5.0, 15.0) == "far"

    def test_is_in_service_box_does_not_require_calibration(self):
        """is_in_service_box is purely geometric, doesn't need a calibrated camera."""
        cam = CameraModel()  # uncalibrated
        assert cam.is_in_service_box(2.5, 8.0, "near_left") is True
        assert cam.is_in_service_box(2.5, 8.0, "near_right") is False


# ---------------------------------------------------------------------------
# Reprojection error
# ---------------------------------------------------------------------------

@pytest.fixture
def calibrated_camera():
    """Alias for calibrated_3d_camera — full 3D model with 12+2 keypoints."""
    cam = CameraModel()
    keypoints_2d = [
        [100, 680],
        [1180, 680],
        [280, 500],
        [640, 500],
        [1000, 500],
        [240, 400],
        [1040, 400],
        [300, 310],
        [640, 310],
        [980, 310],
        [380, 160],
        [900, 160],
    ]
    net_top_2d = [
        [230, 370],
        [1050, 370],
    ]
    cam.calibrate(keypoints_2d, net_top_2d=net_top_2d, image_width=1280, image_height=720)
    return cam


def test_compute_reprojection_error(calibrated_camera):
    # Use the same keypoints used for calibration — reprojection error should be low
    keypoints_2d = [
        [100, 680],
        [1180, 680],
        [280, 500],
        [640, 500],
        [1000, 500],
        [240, 400],
        [1040, 400],
        [300, 310],
        [640, 310],
        [980, 310],
        [380, 160],
        [900, 160],
    ]
    error = calibrated_camera.compute_reprojection_error(keypoints_2d)
    assert error is not None
    assert error >= 0.0
    assert error < 100.0


def test_reprojection_error_uncalibrated():
    cam = CameraModel()
    error = cam.compute_reprojection_error([[100, 200]])
    assert error is None
58 -0
