import numpy as np
import pytest
from cv.court_calibration import CourtCalibration


class TestHomographyCalibration:
    def test_calibrate_from_4_corners(self, sample_court_corners_pixels, court_real_coords):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.homography is not None
        assert cal.homography.shape == (3, 3)

    def test_pixel_to_court_at_corners(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        result = cal.pixel_to_court(320, 700)
        assert abs(result[0] - 0.0) < 0.1
        assert abs(result[1] - 0.0) < 0.1

    def test_pixel_to_court_far_right(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        result = cal.pixel_to_court(1200, 200)
        assert abs(result[0] - 10.0) < 0.1
        assert abs(result[1] - 20.0) < 0.1

    def test_court_to_pixel_roundtrip(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        px, py = cal.court_to_pixel(5.0, 10.0)
        rx, ry = cal.pixel_to_court(px, py)
        assert abs(rx - 5.0) < 0.2
        assert abs(ry - 10.0) < 0.2

    def test_raises_without_calibration(self):
        cal = CourtCalibration()
        with pytest.raises(RuntimeError, match="not calibrated"):
            cal.pixel_to_court(100, 100)

    def test_requires_4_points(self):
        cal = CourtCalibration()
        with pytest.raises(ValueError, match="4 corner points"):
            cal.calibrate(np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32))


class TestCourtZones:
    def test_is_in_bounds(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.is_in_bounds(5.0, 10.0) is True
        assert cal.is_in_bounds(-1.0, 10.0) is False
        assert cal.is_in_bounds(11.0, 10.0) is False
        assert cal.is_in_bounds(5.0, -1.0) is False
        assert cal.is_in_bounds(5.0, 21.0) is False

    def test_get_court_side(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.get_court_side(5.0, 5.0) == "near"
        assert cal.get_court_side(5.0, 15.0) == "far"

    def test_is_in_service_box(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.is_in_service_box(3.0, 8.0, "near_left") is True
        assert cal.is_in_service_box(7.0, 8.0, "near_right") is True
        assert cal.is_in_service_box(3.0, 12.0, "far_left") is True
        assert cal.is_in_service_box(3.0, 5.0, "near_left") is False

    def test_net_line_y(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        assert cal.NET_Y == 10.0


class TestCalibrationPersistence:
    def test_to_dict_and_from_dict(self, sample_court_corners_pixels):
        cal = CourtCalibration()
        cal.calibrate(sample_court_corners_pixels)
        data = cal.to_dict()
        assert "homography" in data
        assert "corners_pixels" in data
        cal2 = CourtCalibration.from_dict(data)
        result = cal2.pixel_to_court(320, 700)
        assert abs(result[0] - 0.0) < 0.1
