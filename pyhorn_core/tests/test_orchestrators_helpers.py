"""
Tests for acoustic power → SPL helpers and the Futtrup GDlimit
in pyhorn_physics.orchestrators.

These helpers were identified as untested in the 2026-05-05 parallel sprint.
"""
import numpy as np
import pytest

from pyhorn_core.pyhorn_physics.orchestrators import (
    acoustic_power_to_spl_dB_W_m,
    _compute_futtrup_gdlimit,
    _pressure_to_spl,
)


# ─── acoustic_power_to_spl_dB_W_m ──────────────────────────────────────────────

class TestAcousticPowerToSpl:
    """acoustic_power_to_spl_dB_W_m converts acoustic power (W) → SPL dB/W/m."""

    def test_scalar_zero_power(self):
        """Zero acoustic power → sensitivity_db (reference level)."""
        result = acoustic_power_to_spl_dB_W_m(0.0, sensitivity_db=89.0)
        assert result == pytest.approx(89.0)

    def test_scalar_known_power(self):
        """P=1e-12 W is the acoustic reference → SPL = sensitivity_db."""
        result = acoustic_power_to_spl_dB_W_m(1e-12, sensitivity_db=0.0)
        assert result == pytest.approx(0.0)

    def test_scalar_1w_acoustic(self):
        """P=1 W → 120 dB re 1e-12 W, plus sensitivity offset."""
        result = acoustic_power_to_spl_dB_W_m(1.0, sensitivity_db=0.0)
        assert result == pytest.approx(120.0)

    def test_scalar_with_sensitivity_offset(self):
        """FE166NV2-style sensitivity offset applied correctly."""
        # P=1 W → 120 dB; with sensitivity_db=-8 → 112 dB
        result = acoustic_power_to_spl_dB_W_m(1.0, sensitivity_db=-8.0)
        assert result == pytest.approx(112.0)

    def test_array_input(self):
        """Array of acoustic powers returns array of SPL values."""
        p = np.array([0.0, 1e-12, 1.0])
        result = acoustic_power_to_spl_dB_W_m(p, sensitivity_db=0.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0)      # 0W → sensitivity (0dB ref)
        assert result[1] == pytest.approx(0.0)      # 1e-12 W → 0dB
        assert result[2] == pytest.approx(120.0)    # 1W → 120dB

    def test_array_with_offset(self):
        """Sensitivity offset applies to every element."""
        p = np.array([1e-12, 1e-6, 1e-3])
        result = acoustic_power_to_spl_dB_W_m(p, sensitivity_db=-10.0)
        # 1e-12 → 0 + (-10) = -10; 1e-6 → 60 + (-10) = 50; 1e-3 → 90 + (-10) = 80
        expected = np.array([0.0, 60.0, 90.0]) + (-10.0)
        np.testing.assert_allclose(result, expected)

    def test_negative_power_clamped(self):
        """Negative power values are clamped to P_REF (1e-12 W)."""
        # -1 W should behave like 1e-12 (clamped), not error
        result = acoustic_power_to_spl_dB_W_m(-1.0, sensitivity_db=0.0)
        assert result == pytest.approx(0.0)


# ─── _compute_futtrup_gdlimit ──────────────────────────────────────────────────

class TestFuttrupGdlimit:
    """_compute_futtrup_gdlimit computes the audible group delay limit curve."""

    def test_returns_numpy_array(self):
        """Should return a numpy array matching the input frequency shape."""
        freqs = np.linspace(20, 5000, 100)
        result = _compute_futtrup_gdlimit(freqs)
        assert isinstance(result, np.ndarray)
        assert result.shape == freqs.shape

    def test_gd_decreases_with_frequency(self):
        """Group delay limit should decrease monotonically with frequency."""
        freqs = np.array([20, 50, 100, 200, 500, 1000, 2000, 5000])
        gd = _compute_futtrup_gdlimit(freqs)
        # Each successive value should be less than or equal (monotone decreasing)
        for i in range(1, len(gd)):
            assert gd[i] <= gd[i - 1], f"GD increased at freq={freqs[i]}"

    def test_positive_values_only(self):
        """GDlimit should never be negative."""
        freqs = np.linspace(20, 20000, 500)
        gd = _compute_futtrup_gdlimit(freqs)
        assert np.all(gd > 0.0)

    def test_very_low_frequency_bounded(self):
        """Below ~50 Hz the denominator in the formula approaches zero;
        values are clamped to a safe upper bound."""
        freqs = np.array([10.0, 20.0, 30.0])
        gd = _compute_futtrup_gdlimit(freqs)
        # All values should be finite and positive
        assert np.all(np.isfinite(gd))
        assert np.all(gd > 0.0)
        # Upper clamped value: 1000 * 1160.6 / 1.0 = 1_160_600 ms (safe ceiling)
        assert np.all(gd <= 1_161_000)

    def test_high_frequency_small_gd(self):
        """At high frequencies GDlimit should be small (< 1 ms above ~1 kHz)."""
        freqs = np.array([1000.0, 2000.0, 5000.0])
        gd = _compute_futtrup_gdlimit(freqs)
        assert np.all(gd < 1.0), f"GDlimit at HF should be < 1 ms: {gd}"

    def test_known_value_1khz(self):
        """Spot-check: f=1000 Hz — empirically ~0.74 ms."""
        freqs = np.array([1000.0])
        gd = _compute_futtrup_gdlimit(freqs)
        assert gd[0] == pytest.approx(0.738, abs=0.001)

    def test_known_value_100hz(self):
        """Spot-check: f=100 Hz — empirically ~4.82 ms."""
        freqs = np.array([100.0])
        gd = _compute_futtrup_gdlimit(freqs)
        assert gd[0] == pytest.approx(4.82, abs=0.01)


# ─── _pressure_to_spl ─────────────────────────────────────────────────────────

class TestPressureToSpl:
    """_pressure_to_spl converts acoustic pressure (Pa) → SPL in dB SPL.

    Reference: SPL = 20·log10(|p| / 2e-5) dB.  A small floor (1e-12) is added
    inside the log to prevent log(0) when pressure is exactly zero.
    """

    def test_reference_pressure(self):
        """p = 2e-5 Pa (reference) → 0 dB SPL."""
        result = _pressure_to_spl(np.array([2e-5]))[0]
        assert result == pytest.approx(0.0, abs=0.01)

    def test_10x_reference(self):
        """10× reference pressure → +20 dB SPL (log property)."""
        result = _pressure_to_spl(np.array([2e-4]))[0]
        assert result == pytest.approx(20.0, abs=0.01)

    def test_zero_pressure_floored(self):
        """Zero pressure → floor term (1e-12) is added before log, giving -240 dB.

        Formula: 20·log10(0 + 1e-12) = 20·log10(1e-12) = 20·(−12) = −240 dB SPL.
        """
        result = _pressure_to_spl(np.array([0.0]))[0]
        assert result == pytest.approx(-240.0, abs=1.0)

    def test_complex_pressure_uses_magnitude(self):
        """Complex pressure (e.g. phasor) should use its absolute value."""
        p = 1.0 + 1.0j  # |p| = sqrt(2)
        result = _pressure_to_spl(np.array([p]))[0]
        expected = 20.0 * np.log10(np.sqrt(2) / 2e-5)
        assert result == pytest.approx(expected, abs=0.01)

    def test_array_of_pressures(self):
        """Array input returns array of SPL values."""
        p = np.array([2e-5, 2e-4, 2e-3])  # 0, +20, +40 dB SPL
        result = _pressure_to_spl(p)
        assert result.shape == (3,)
        assert result[0] == pytest.approx(0.0, abs=0.01)
        assert result[1] == pytest.approx(20.0, abs=0.01)
        assert result[2] == pytest.approx(40.0, abs=0.01)


# ─── _detect_numerical_artifacts ──────────────────────────────────────────────

from pyhorn_core.pyhorn_physics.orchestrators import (
    _detect_numerical_artifacts,
)


class TestDetectNumericalArtifacts:
    """Tests for _detect_numerical_artifacts()."""

    def test_flat_spl_returns_empty(self):
        """A flat (constant) SPL array has no artifacts."""
        freqs = np.linspace(100, 5000, 100)
        spl = np.full_like(freqs, 90.0)
        result = _detect_numerical_artifacts(freqs, spl)
        assert result == []

    def test_smooth_gradient_returns_empty(self):
        """A smooth SPL rolloff with no sharp spikes returns no artifacts."""
        freqs = np.linspace(100, 5000, 200)
        spl = 100.0 - 5.0 * np.log10(freqs / 100.0)  # smooth -5 dB/log decade
        result = _detect_numerical_artifacts(freqs, spl)
        assert result == []

    def test_isolated_spike_detected(self):
        """A single isolated 30 dB spike in an otherwise flat SPL is detected."""
        freqs = np.linspace(100, 5000, 100)
        spl = np.full_like(freqs, 90.0)
        # spike at index 50 → freq ≈ 2500 Hz
        spl[50] = 120.0
        result = _detect_numerical_artifacts(freqs, spl)
        # Should contain the spike frequency
        assert len(result) >= 1
        assert any(abs(f - freqs[50]) < 50.0 for f in result)

    def test_trend_break_detected(self):
        """A sudden trend break (large deviation from local median) is detected."""
        freqs = np.linspace(100, 5000, 200)
        spl = np.linspace(100.0, 80.0, 200)  # smooth rolloff
        # Abrupt dip of 30 dB at index 100 → unphysical
        spl[100] = 40.0
        result = _detect_numerical_artifacts(freqs, spl)
        assert len(result) >= 1

    def test_too_few_points_returns_empty(self):
        """Arrays with fewer than 5 points return empty (guard against median crash)."""
        freqs = np.array([100.0, 200.0, 300.0])
        spl = np.array([90.0, 100.0, 95.0])
        result = _detect_numerical_artifacts(freqs, spl)
        assert result == []

    def test_multiple_artifacts(self):
        """Multiple spikes at different frequencies are all detected."""
        freqs = np.linspace(100, 10000, 500)
        spl = np.full_like(freqs, 90.0)
        # Two isolated spikes
        spike_indices = [100, 300]
        for idx in spike_indices:
            spl[idx] = 125.0
        result = _detect_numerical_artifacts(freqs, spl)
        assert len(result) >= 2

    def test_result_sorted_ascending(self):
        """Returned artifact frequencies are sorted in ascending order."""
        freqs = np.linspace(100, 10000, 500)
        spl = np.full_like(freqs, 90.0)
        spl[50] = 120.0
        spl[400] = 115.0
        result = _detect_numerical_artifacts(freqs, spl)
        assert result == sorted(result)

    def test_small_oscillation_not_flagged(self):
        """Small oscillatory SPL variations (normal response structure) are not flagged.

        A typical horn SPL curve may have ±5 dB ripples. These are physical, not
        numerical artifacts, and should NOT be detected (threshold is 20 dB).
        """
        freqs = np.linspace(100, 5000, 200)
        # ±5 dB ripple — below the 20 dB threshold
        spl = 90.0 + 5.0 * np.sin(2 * np.pi * freqs / 500.0)
        result = _detect_numerical_artifacts(freqs, spl)
        assert result == []
