"""Unit tests for pyhorn.config.models — DriverSpecs and HornGeometry dataclasses."""

import math
import pytest
from pyhorn_core.config.models import DriverSpecs, HornGeometry


class TestDriverSpecs:
    """Tests for DriverSpecs Thiele-Small parameter dataclass."""

    def test_valid_construction(self):
        """DriverSpecs should construct with required fields only."""
        driver = DriverSpecs(
            fs=50.0,
            qts=0.27,
            qes=0.28,
            qms=7.5,
            vas=0.0369,
            re=7.8,
            bl=7.79,
            mms=0.007,
            cms=1.472e-3,
            rms=0.277,
            sd=0.01327,
        )
        assert driver.fs == 50.0
        assert driver.qts == 0.27
        assert driver.re == 7.8

    def test_defaults(self):
        """DriverSpecs should apply correct defaults for optional fields."""
        driver = DriverSpecs(
            fs=50.0, qts=0.27, qes=0.28, qms=7.5, vas=0.0369,
            re=7.8, bl=7.79, mms=0.007, cms=1.472e-3, rms=0.277, sd=0.01327,
        )
        assert driver.voltage == 2.83
        assert driver.le == 0.0
        assert driver.xmax == 0.0

    def test_reference_spl_positive(self):
        """reference_spl should return a positive value for a typical driver."""
        driver = DriverSpecs(
            fs=50.0, qts=0.27, qes=0.28, qms=7.5, vas=0.0369,
            re=7.8, bl=7.79, mms=0.007, cms=1.472e-3, rms=0.277, sd=0.01327,
        )
        spl = driver.reference_spl
        assert spl > 0

    def test_reference_spl_increases_with_voltage(self):
        """reference_spl should increase when voltage is raised."""
        driver_low = DriverSpecs(
            fs=50.0, qts=0.27, qes=0.28, qms=7.5, vas=0.0369,
            re=7.8, bl=7.79, mms=0.007, cms=1.472e-3, rms=0.277, sd=0.01327,
            voltage=2.83,
        )
        driver_high = DriverSpecs(
            fs=50.0, qts=0.27, qes=0.28, qms=7.5, vas=0.0369,
            re=7.8, bl=7.79, mms=0.007, cms=1.472e-3, rms=0.277, sd=0.01327,
            voltage=10.0,
        )
        assert driver_high.reference_spl > driver_low.reference_spl


class TestHornGeometryDefaults:
    """Tests for HornGeometry defaults."""

    def test_enclosure_type_default_flh(self):
        """Default enclosure_type should be 'FLH'."""
        hg = HornGeometry()
        assert hg.enclosure_type == "FLH"

    def test_ang_default_two_pi(self):
        """Default ang should be 2*pi (full steradians)."""
        hg = HornGeometry()
        assert hg.ang == pytest.approx(6.283185307, rel=1e-9)

    def test_n_segments_default_100(self):
        """Default n_segments should be 100."""
        hg = HornGeometry()
        assert hg.n_segments == 100

    def test_conical_segments_default_none(self):
        """conical_segments should default to None."""
        hg = HornGeometry()
        assert hg.conical_segments is None

    def test_rectangular_segments_default_none(self):
        """rectangular_segments should default to None."""
        hg = HornGeometry()
        assert hg.rectangular_segments is None

    def test_segments_default_empty_list(self):
        """segments should default to empty list."""
        hg = HornGeometry()
        assert hg.segments == []

    def test_width_default_none(self):
        """width should default to None."""
        hg = HornGeometry()
        assert hg.width is None


class TestHornGeometryGeometryDiagnostics:
    """Tests for HornGeometry.geometry_diagnostics()."""

    def test_diagnostics_empty_geometry(self):
        """Empty geometry should return empty dict."""
        hg = HornGeometry()
        assert hg.geometry_diagnostics() == {}

    def test_diagnostics_conical_segments_with_width(self):
        """conical_segments with width set should compute diagnostics correctly."""
        hg = HornGeometry(
            width=0.2,
            conical_segments=[
                (0.05, 0.06, 0.03),
                (0.06, 0.07, 0.04),
            ],
        )
        d = hg.geometry_diagnostics()
        assert d["segment_count"] == 2.0
        assert d["min_segment_length_m"] == 0.03
        assert d["max_segment_length_m"] == 0.04
        # Area = height * width = 0.05 * 0.2 = 0.01
        assert d["min_area_m2"] == pytest.approx(0.05 * 0.2)
        assert d["max_area_m2"] == pytest.approx(0.07 * 0.2)

    def test_diagnostics_conical_segments_max_area_ratio(self):
        """max_area_step_ratio reflects adjacent end-to-start area ratios.
        
        Segments transition: seg1_end→seg2_start ratio should be computed.
        e.g. seg1 ends at area 0.07, seg2 starts at area 0.08 → ratio 8/7 ≈ 1.143.
        """
        # Segments: (height_start, height_end, length)
        # With width=0.2:
        #   seg1: 0.05 → 0.07 (area 0.010 → 0.014)
        #   seg2: 0.08 → 0.09 (area 0.016 → 0.018)
        #   seg3: 0.08 → 0.10 (area 0.016 → 0.020)
        # Transition seg1→seg2: max(0.014,0.016)/min(0.014,0.016) = 0.016/0.014 = 8/7 ≈ 1.143
        # Transition seg2→seg3: max(0.018,0.016)/min(0.018,0.016) = 0.018/0.016 = 1.125
        # max_area_step_ratio = 8/7 ≈ 1.143
        hg = HornGeometry(
            width=0.2,
            conical_segments=[
                (0.05, 0.07, 0.03),
                (0.08, 0.09, 0.04),
                (0.08, 0.10, 0.035),
            ],
        )
        d = hg.geometry_diagnostics()
        assert d["max_area_step_ratio"] == pytest.approx(8.0 / 7.0, rel=1e-9)

    def test_diagnostics_rectangular_segments(self):
        """rectangular_segments should produce correct diagnostics."""
        hg = HornGeometry(
            rectangular_segments=[
                (0.1, 0.05, 0.1, 0.06, 0.03),   # w, h_start, w, h_end, length
                (0.1, 0.06, 0.1, 0.08, 0.04),
            ],
        )
        d = hg.geometry_diagnostics()
        assert d["segment_count"] == 2.0
        assert d["min_width_m"] == 0.1
        assert d["max_width_m"] == 0.1
        assert d["min_height_m"] == 0.05
        assert d["max_height_m"] == 0.08

    def test_diagnostics_bend_angles(self):
        """bend_angles should contribute bend diagnostics."""
        hg = HornGeometry(
            bend_angles=[math.radians(30), math.radians(60), math.radians(45)],
        )
        d = hg.geometry_diagnostics()
        assert d["max_bend_angle_deg"] == pytest.approx(60.0, abs=1e-6)
        assert d["mean_bend_angle_deg"] == pytest.approx(45.0, abs=1e-6)

    def test_diagnostics_lem_enabled(self):
        """lem_step_model set to 'basic' should set lem_enabled=1.0."""
        hg = HornGeometry(lem_step_model="basic")
        d = hg.geometry_diagnostics()
        assert d["lem_enabled"] == 1.0

    def test_diagnostics_lem_disabled_for_ideal(self):
        """lem_step_model='ideal' should NOT set lem_enabled."""
        hg = HornGeometry(lem_step_model="ideal")
        d = hg.geometry_diagnostics()
        assert "lem_enabled" not in d


class TestHornGeometryFoldedPlotSegments:
    """Tests for HornGeometry.folded_plot_segments()."""

    def test_conical_returns_conical_segments(self):
        """For conical_segments, folded_plot_segments should return them."""
        segs = [(0.05, 0.06, 0.03), (0.06, 0.07, 0.04)]
        hg = HornGeometry(conical_segments=segs)
        assert hg.folded_plot_segments() == segs

    def test_rectangular_returns_height_triples(self):
        """For rectangular_segments, return (h_start, h_end, length) triples."""
        segs = [(0.1, 0.05, 0.1, 0.06, 0.03)]
        hg = HornGeometry(rectangular_segments=segs)
        result = hg.folded_plot_segments()
        assert result == [(0.05, 0.06, 0.03)]

    def test_no_segments_returns_none(self):
        """When neither conical nor rectangular segments are set, return None."""
        hg = HornGeometry()
        assert hg.folded_plot_segments() is None