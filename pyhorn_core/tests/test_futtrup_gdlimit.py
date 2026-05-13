"""Unit tests for the Futtrup audible group delay limit formula.

Reference: Hornresp page 113 — Futtrup formula:
    GDlimit = 1000 × 1160.6 / (5643 × f^0.81511 − f)  [ms]
"""

import numpy as np
import pytest
from pathlib import Path
from pyhorn_core.pyhorn_physics.orchestrators import (
    _compute_futtrup_gdlimit,
    horn_response,
)
from pyhorn_core.config.parser import parse_driver_specs


class TestFuttrupGDlimit:
    """Tests for _compute_futtrup_gdlimit()."""

    @pytest.fixture
    def freqs(self):
        """Log-spaced frequencies from 20 Hz to 5 kHz."""
        return np.logspace(np.log10(20), np.log10(5000), 500)

    def test_returns_numpy_array(self, freqs):
        """Should return a numpy array of the same shape as freqs."""
        result = _compute_futtrup_gdlimit(freqs)
        assert isinstance(result, np.ndarray)
        assert result.shape == freqs.shape

    def test_decreases_with_frequency(self, freqs):
        """GD limit should monotonically decrease with increasing frequency.

        The Futtrup formula gives a higher GD limit at low frequencies,
        reflecting the ear's greater temporal integration window.
        """
        gd = _compute_futtrup_gdlimit(freqs)
        # Check monotonic decrease: each subsequent value should be <= the previous
        diffs = np.diff(gd)
        # Allow tiny numerical noise (tol=1e-6 ms)
        assert np.all(diffs <= 1e-6), (
            f"Futtrup GD limit is not monotonically decreasing. "
            f"Found {np.sum(diffs > 1e-6)} positive steps."
        )

    def test_low_frequency_range_4_to_20ms(self, freqs):
        """At low frequencies (30–100 Hz) the GD limit should be in the 4–20 ms range.

        Futtrup's measurements show audible GD is bounded by roughly this window
        at low frequencies. At 100 Hz the formula gives ~4.8 ms (approaching the
        floor); at 30 Hz it gives ~13 ms. Values outside 4–20 ms would indicate
        a formula error.
        """
        gd = _compute_futtrup_gdlimit(freqs)
        low_freq_mask = (freqs >= 30) & (freqs <= 100)
        low_freq_gd = gd[low_freq_mask]
        assert np.all(low_freq_gd >= 4.0), (
            f"Low-frequency GD limit below 4 ms: {low_freq_gd.min():.2f} ms"
        )
        assert np.all(low_freq_gd <= 20.0), (
            f"Low-frequency GD limit above 20 ms: {low_freq_gd.max():.2f} ms"
        )

    def test_high_frequency_limit_small(self, freqs):
        """At high frequencies (1–5 kHz) the GD limit should be very small (< 2 ms)."""
        gd = _compute_futtrup_gdlimit(freqs)
        high_freq_mask = (freqs >= 1000) & (freqs <= 5000)
        high_freq_gd = gd[high_freq_mask]
        assert np.all(high_freq_gd <= 2.0), (
            f"High-frequency GD limit above 2 ms: {high_freq_gd.max():.3f} ms"
        )

    def test_clamped_below_50hz(self, freqs):
        """Below ~50 Hz the denominator approaches zero; result should be clamped
        to a finite upper bound (no division-by-zero or overflow).
        """
        gd = _compute_futtrup_gdlimit(freqs)
        very_low_mask = freqs < 50
        very_low_gd = gd[very_low_mask]
        # Should be finite and bounded
        assert np.all(np.isfinite(very_low_gd)), "Non-finite GD limit values found"
        assert np.all(very_low_gd <= 1e5), (
            f"GD limit not properly clamped below 50 Hz: {very_low_gd.max():.1f} ms"
        )

    def test_formula_numerical_check(self):
        """Spot-check the formula at a specific frequency.

        At f = 100 Hz:
          denominator = 5643 * 100^0.81511 - 100
          GDlimit = 1000 * 1160.6 / denominator
        """
        f_test = 100.0
        denom = 5643.0 * f_test**0.81511 - f_test
        expected = 1000.0 * 1160.6 / denom
        result = _compute_futtrup_gdlimit(np.array([f_test]))[0]
        assert result == pytest.approx(expected, rel=1e-9)

    def test_matches_horn_response_result(self, freqs):
        """_compute_futtrup_gdlimit should match result.futtrup_gdlimit from horn_response."""
        # Use a minimal driver + horn to exercise the path
        driver_data = {
            "re": 6.0,
            "bl": 5.0,
            "fs": 45.0,
            "qts": 0.38,
            "qes": 0.50,
            "qms": 2.50,
            "vas": 30.0,
            "mms": 0.010,
            "cms": 0.0005,
            "rms": 0.5,
            "sd": 0.015,
            "le": 0.0003,
            "le_freq_dependency": False,
            "lossy_le": False,
            "voltage": 2.0,
            "xmax": 3.0,
        }
        import tempfile, yaml
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(driver_data, f)
            driver_path = f.name
        driver = parse_driver_specs(Path(driver_path))

        from pyhorn_core.config.models import HornGeometry
        horn = HornGeometry(
            enclosure_type="BLH",
            throat_area=20.0,
            mouth_area=300.0,
            path_length=1.5,
            n_segments=20,
            vrc=0.0,
            lrc=0.0,
            fr_rc=0.0,
            ap1=0.0,
            lpt=0.0,
            ang=2.0 * np.pi,
        )

        result = horn_response(freqs[:50], driver, horn, compute_distortion=False)

        direct = _compute_futtrup_gdlimit(freqs[:50])
        np.testing.assert_allclose(
            result.futtrup_gdlimit, direct, rtol=1e-10,
            err_msg="horn_response().futtrup_gdlimit differs from _compute_futtrup_gdlimit"
        )
