"""Tests for pyhorn.solver.folding — horn folding and layout planning."""

import math
import pytest

from pyhorn_core.config.models import HornGeometry
from pyhorn_fold import (
    _point_along_segment,
    throat_chamber_side_length,
    _throat_attachment_point,
    extrapolate_folded_horn,
    _segment_height_limit,
)


# ─── _point_along_segment ───────────────────────────────────────────────────────

class TestPointAlongSegment:
    def test_midpoint(self):
        result = _point_along_segment((0.0, 0.0), (10.0, 0.0), 5.0)
        assert result == (5.0, 0.0)

    def test_fractional_distance(self):
        result = _point_along_segment((0.0, 0.0), (10.0, 0.0), 2.5)
        assert result == (2.5, 0.0)

    def test_diagonal_segment(self):
        result = _point_along_segment((0.0, 0.0), (10.0, 10.0), math.sqrt(200))
        assert result == pytest.approx((10.0, 10.0))

    def test_zero_length_segment_returns_start(self):
        result = _point_along_segment((1.0, 2.0), (1.0, 2.0), 5.0)
        assert result == (1.0, 2.0)


# ─── throat_chamber_side_length ────────────────────────────────────────────────

class TestThroatChamberSideLength:
    def _horn(self, vtc: float) -> HornGeometry:
        return HornGeometry(vtc=vtc)

    def test_positive_volume_and_width_returns_sqrt_of_area(self):
        # vtc=0.0088, enclosure_width=0.18
        # chamber_volume = 0.0088+0.001 = 0.0098
        # chamber_area_2d = 0.0098/0.18 ≈ 0.05444
        # side = sqrt(0.05444) ≈ 0.233
        horn = self._horn(vtc=0.0088)
        result = throat_chamber_side_length(horn, enclosure_width=0.18)
        expected = math.sqrt((0.0088 + 1e-3) / 0.18)
        assert result == pytest.approx(expected, abs=1e-4)

    def test_negative_vtc_returns_zero(self):
        # vtc=-1e-3 → chamber_volume = -1e-3 + 1e-3 = 0 → returns 0.0
        horn = self._horn(vtc=-1e-3)
        result = throat_chamber_side_length(horn, enclosure_width=0.18)
        assert result == 0.0

    def test_negative_vtc_returns_zero(self):
        horn = self._horn(vtc=-0.001)
        result = throat_chamber_side_length(horn, enclosure_width=0.18)
        assert result == 0.0

    def test_zero_enclosure_width_returns_zero(self):
        horn = self._horn(vtc=0.0088)
        result = throat_chamber_side_length(horn, enclosure_width=0.0)
        assert result == 0.0

    def test_negative_enclosure_width_returns_zero(self):
        horn = self._horn(vtc=0.0088)
        result = throat_chamber_side_length(horn, enclosure_width=-0.1)
        assert result == 0.0


# ─── _throat_attachment_point ─────────────────────────────────────────────────

class TestThroatAttachmentPoint:
    def test_zero_chamber_side_returns_driver_coord(self):
        result = _throat_attachment_point(
            enclosure_dims=(0.34, 0.61),
            driver_coord=(0.05, 0.3),
            chamber_side=0.0,
        )
        assert result == (0.05, 0.3)

    def test_driver_on_left_half(self):
        result = _throat_attachment_point(
            enclosure_dims=(0.34, 0.61),
            driver_coord=(0.05, 0.3),
            chamber_side=0.1,
        )
        # chamber_x0 = 0, chamber centered at x=0.05
        assert result[0] == pytest.approx(0.05)  # center of chamber at x=0.05

    def test_driver_on_right_half(self):
        result = _throat_attachment_point(
            enclosure_dims=(0.34, 0.61),
            driver_coord=(0.3, 0.3),
            chamber_side=0.1,
        )
        # driver on right half → chamber_x0 = depth - chamber_side
        # chamber at x = 0.34 - 0.05 = 0.29
        assert result[0] == pytest.approx(0.29)

    def test_chamber_side_clamped_to_depth(self):
        result = _throat_attachment_point(
            enclosure_dims=(0.2, 0.61),
            driver_coord=(0.1, 0.3),
            chamber_side=0.5,  # larger than depth
        )
        # chamber_side clamped to min(depth, height) = 0.2
        # chamber_x0 = 0.0, center at x = 0.0 + 0.1 = 0.1
        assert result[0] == pytest.approx(0.1)


# ─── extrapolate_folded_horn — validation errors ────────────────────────────────

class TestExtrapolateFoldedHornErrors:
    def _horn(self, **kwargs) -> HornGeometry:
        defaults = dict(
            profile_type="Exponential",
            throat_area=0.005,
            mouth_area=0.05,
            path_length=1.0,
            vtc=0.0,
        )
        defaults.update(kwargs)
        return HornGeometry(**defaults)

    def test_profile_type_none_raises(self):
        horn = self._horn(profile_type=None)
        with pytest.raises(ValueError, match="continuous profile"):
            extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)

    def test_zero_depth_raises(self):
        horn = self._horn()
        with pytest.raises(ValueError, match="Enclosure dimensions"):
            extrapolate_folded_horn(horn, (0.0, 0.61), (0.05, 0.3), 0.18)

    def test_negative_height_raises(self):
        horn = self._horn()
        with pytest.raises(ValueError, match="Enclosure dimensions"):
            extrapolate_folded_horn(horn, (0.34, -0.5), (0.05, 0.3), 0.18)

    def test_zero_enclosure_width_raises(self):
        horn = self._horn()
        with pytest.raises(ValueError, match="Enclosure width"):
            extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.0)

    def test_negative_enclosure_width_raises(self):
        horn = self._horn()
        with pytest.raises(ValueError, match="Enclosure width"):
            extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), -0.1)

    def test_driver_x_outside_enclosure_raises(self):
        horn = self._horn()
        with pytest.raises(ValueError, match="driver_x must lie"):
            extrapolate_folded_horn(horn, (0.34, 0.61), (0.5, 0.3), 0.18)

    def test_driver_y_too_low_raises(self):
        horn = self._horn()
        with pytest.raises(ValueError, match="driver_y must lie"):
            extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.005), 0.18)

    def test_driver_y_too_high_raises(self):
        horn = self._horn()
        with pytest.raises(ValueError, match="driver_y must lie"):
            extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.6), 0.18)

    def test_throat_chamber_exceeds_enclosure_height_raises(self):
        # Large vtc → large throat chamber → may not fit in enclosure
        horn = self._horn(vtc=0.1)  # large throat chamber
        with pytest.raises(ValueError, match="throat chamber"):
            extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)


# ─── extrapolate_folded_horn — successful cases ────────────────────────────────

class TestExtrapolateFoldedHornSuccess:
    def _horn(self, **kwargs) -> HornGeometry:
        defaults = dict(
            profile_type="Exponential",
            throat_area=0.005,
            mouth_area=0.05,
            path_length=1.0,
            vtc=0.0,
        )
        defaults.update(kwargs)
        return HornGeometry(**defaults)

    def test_returns_horn_geometry(self):
        horn = self._horn()
        result = extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)
        assert isinstance(result, HornGeometry)

    def test_result_has_rectangular_segments(self):
        horn = self._horn()
        result = extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)
        assert result.rectangular_segments is not None
        assert len(result.rectangular_segments) > 0

    def test_result_has_coordinates(self):
        horn = self._horn()
        result = extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)
        assert result.coordinates is not None
        assert len(result.coordinates) > 0

    def test_result_has_enclosure_dims(self):
        horn = self._horn()
        result = extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)
        assert result.enclosure_dims == (0.34, 0.61)

    def test_hyperbolic_profile_type(self):
        horn = self._horn(profile_type="Hyperbolic", hyperbolic_t=0.7)
        result = extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)
        assert result.rectangular_segments is not None
        assert len(result.rectangular_segments) > 0

    def test_conical_profile_type(self):
        horn = self._horn(profile_type="Conical")
        result = extrapolate_folded_horn(horn, (0.34, 0.61), (0.05, 0.3), 0.18)
        assert result.rectangular_segments is not None
        assert len(result.rectangular_segments) > 0

    def test_driver_coord_carried_through(self):
        horn = self._horn()
        driver_coord = (0.05, 0.3)
        result = extrapolate_folded_horn(horn, (0.34, 0.61), driver_coord, 0.18)
        assert result.driver_coord == driver_coord


# ─── _segment_height_limit ─────────────────────────────────────────────────────

class TestSegmentHeightLimit:
    def test_horizontal_segment_uses_x_boundary(self):
        segment = ((0.1, 0.1), (0.2, 0.1), 0.1)  # horizontal (y0 == y1)
        enclosure_dims = (0.34, 0.61)
        result = _segment_height_limit(segment, enclosure_dims, fill_factor=0.8)
        # y0=0.1, height=0.61 → 2*min(0.1, 0.51) = 0.2
        expected = 0.8 * 0.2
        assert result == pytest.approx(expected)

    def test_vertical_segment_uses_y_boundary(self):
        segment = ((0.1, 0.05), (0.1, 0.15), 0.1)  # vertical (y1 != y0)
        enclosure_dims = (0.34, 0.61)
        result = _segment_height_limit(segment, enclosure_dims, fill_factor=0.8)
        # x0=0.1, depth=0.34 → 2*min(0.1, 0.24) = 0.2
        expected = 0.8 * 0.2
        assert result == pytest.approx(expected)

    def test_zero_fill_factor_returns_zero(self):
        # fill_factor=0.0 → result = 0.0 * max(limit, 1e-6) = 0.0
        segment = ((0.1, 0.05), (0.1, 0.15), 0.1)  # vertical
        enclosure_dims = (0.34, 0.61)
        result = _segment_height_limit(segment, enclosure_dims, fill_factor=0.0)
        assert result == 0.0
