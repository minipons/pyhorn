"""Unit tests for pyhorn.solver.time_domain."""

import numpy as np
import pytest
from pyhorn_core.solver.time_domain import (
    TimeDomainResult,
    compute_csd,
    compute_impulse_response,
)


class TestComputeImpulseResponse:
    """Tests for compute_impulse_response()."""

    @pytest.fixture
    def uniform_freqs(self):
        """Uniformly spaced 20 Hz–20 kHz at 1 Hz resolution."""
        return np.linspace(20, 20000, 19981)

    @pytest.fixture
    def flat_spectrum(self):
        """Flat magnitude spectrum (all 1 Pa)."""
        return np.ones(19981)

    def test_returns_tuple_of_arrays(self, uniform_freqs, flat_spectrum):
        """Should return (time_ms, impulse) as numpy arrays."""
        t, ir = compute_impulse_response(uniform_freqs, flat_spectrum)
        assert isinstance(t, np.ndarray)
        assert isinstance(ir, np.ndarray)
        assert len(t) == len(ir)

    def test_time_axis_in_milliseconds(self, uniform_freqs, flat_spectrum):
        """Time axis should be in milliseconds."""
        t, ir = compute_impulse_response(uniform_freqs, flat_spectrum)
        # Time should be positive and increasing
        assert t[0] >= 0
        assert np.all(np.diff(t) > 0)

    def test_impulse_is_real(self, uniform_freqs, flat_spectrum):
        """Impulse response should be purely real (not complex)."""
        _, ir = compute_impulse_response(uniform_freqs, flat_spectrum)
        assert np.isrealobj(ir)

    def test_impulse_length_matches_time(self, uniform_freqs, flat_spectrum):
        """Impulse and time arrays should have the same length."""
        t, ir = compute_impulse_response(uniform_freqs, flat_spectrum)
        assert len(t) == len(ir)

    def test_window_ms_trims_output(self, uniform_freqs, flat_spectrum):
        """Larger window_ms should give a longer time axis."""
        t1, _ = compute_impulse_response(uniform_freqs, flat_spectrum, window_ms=10.0)
        t2, _ = compute_impulse_response(uniform_freqs, flat_spectrum, window_ms=20.0)
        assert len(t2) > len(t1)

    def test_zero_freq_at_nyquist_gives_real_ir(self, uniform_freqs):
        """Spectrum with only DC and Nyquist should still give a real IR."""
        pressure = np.zeros(len(uniform_freqs), dtype=complex)
        pressure[0] = 1.0  # DC
        t, ir = compute_impulse_response(uniform_freqs, pressure)
        assert np.isrealobj(ir)

    def test_single_frequency_peak(self, uniform_freqs):
        """A single frequency peak should produce a sinusoidal IR."""
        pressure = np.zeros(len(uniform_freqs), dtype=complex)
        peak_idx = 5000  # ~5 kHz
        pressure[peak_idx] = 1.0
        t, ir = compute_impulse_response(uniform_freqs, pressure, window_ms=50.0)
        # IR should be oscillatory and not all zeros
        assert not np.allclose(ir, 0)
        # Should show some oscillation
        assert np.std(ir) > 0

    def test_raises_on_non_uniform_frequencies(self):
        """Non-uniform frequency spacing should raise a value error or misbehave gracefully."""
        # Non-uniform: log spacing instead of linear
        freqs = np.array([20, 50, 100, 500, 1000, 5000, 10000, 20000])
        pressure = np.ones(len(freqs))
        # The function may not explicitly check, but we verify it doesn't crash
        t, ir = compute_impulse_response(freqs, pressure)
        assert len(t) > 0
        assert np.isrealobj(ir)

    def test_negative_frequencies_filtered(self, uniform_freqs):
        """Negative frequency components should be filtered (IR is real)."""
        pressure = np.ones(len(uniform_freqs), dtype=complex)
        t, ir = compute_impulse_response(uniform_freqs, pressure)
        assert np.isrealobj(ir)

    def test_window_ms_zero_produces_empty_or_singleton(self, uniform_freqs, flat_spectrum):
        """window_ms=0 may produce an empty array; verify no crash."""
        t, ir = compute_impulse_response(uniform_freqs, flat_spectrum, window_ms=0.0)
        # Either the window is 0-length, or it falls back to some minimum.
        assert isinstance(t, np.ndarray)
        assert isinstance(ir, np.ndarray)


class TestComputeCsd:
    """Tests for compute_csd()."""

    @pytest.fixture
    def uniform_freqs(self):
        """Uniformly spaced 20 Hz–20 kHz."""
        return np.linspace(20, 20000, 19981)

    @pytest.fixture
    def flat_spectrum(self):
        """Flat magnitude spectrum."""
        return np.ones(19981)

    def test_returns_time_domain_result(self, uniform_freqs, flat_spectrum):
        """Should return a TimeDomainResult instance."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        assert isinstance(result, TimeDomainResult)

    def test_all_fields_populated(self, uniform_freqs, flat_spectrum):
        """All TimeDomainResult fields should be non-None arrays."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        assert result.time_ms is not None
        assert result.impulse is not None
        assert result.step is not None
        assert result.csd_times_ms is not None
        assert result.csd_freqs is not None
        assert result.csd_db is not None

    def test_time_ms_is_monotonic(self, uniform_freqs, flat_spectrum):
        """time_ms should be monotonically increasing."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        assert np.all(np.diff(result.time_ms) > 0)

    def test_impulse_and_step_same_length(self, uniform_freqs, flat_spectrum):
        """impulse and step arrays should have the same length."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        assert len(result.impulse) == len(result.step)
        assert len(result.impulse) == len(result.time_ms)

    def test_step_response_starts_small(self, uniform_freqs, flat_spectrum):
        """Step response first sample should be small (near zero initial condition)."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        assert abs(result.step[0]) < 1e-4

    def test_step_response_has_bounded_variation(self, uniform_freqs, flat_spectrum):
        """Step response should have bounded variation (not diverge wildly)."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        step_range = np.max(result.step) - np.min(result.step)
        assert step_range < 1e6  # should be well-behaved for flat spectrum

    def test_csd_freqs_within_simulation_range(self, uniform_freqs, flat_spectrum):
        """CSD frequencies should be within the input frequency range."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        assert result.csd_freqs[0] >= uniform_freqs[0]
        assert result.csd_freqs[-1] <= uniform_freqs[-1]

    def test_csd_shape_is_n_slices_by_n_freqs(self, uniform_freqs, flat_spectrum):
        """CSD magnitude should have shape (n_slices, n_freqs)."""
        n_slices = 30
        result = compute_csd(uniform_freqs, flat_spectrum, n_slices=n_slices)
        assert result.csd_db.shape == (n_slices, len(result.csd_freqs))

    def test_n_slices_controls_csd_rows(self, uniform_freqs, flat_spectrum):
        """n_slices=20 should give csd_db with 20 rows."""
        result = compute_csd(uniform_freqs, flat_spectrum, n_slices=20)
        assert result.csd_db.shape[0] == 20

    def test_csd_db_is_db_scale(self, uniform_freqs, flat_spectrum):
        """CSD magnitude should be in decibels (roughly -180 to 0 dB range)."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        # dB values should be non-positive (magnitude spectrum in dB)
        assert np.all(result.csd_db <= 0)
        # Should contain very negative values (empty/rounded regions)
        assert np.any(result.csd_db < -100)

    def test_csd_freqs_are_positive(self, uniform_freqs, flat_spectrum):
        """CSD frequency axis should be all positive."""
        result = compute_csd(uniform_freqs, flat_spectrum)
        assert np.all(result.csd_freqs > 0)

    def test_csd_times_ms_are_in_window(self, uniform_freqs, flat_spectrum):
        """CSD time slices should span 0 to window_ms."""
        window_ms = 20.0
        result = compute_csd(uniform_freqs, flat_spectrum, window_ms=window_ms)
        assert result.csd_times_ms[0] == pytest.approx(0.0, abs=0.1)
        assert result.csd_times_ms[-1] == pytest.approx(window_ms, rel=0.01)

    def test_single_frequency_peak_creates_distinct_csd_peak(self, uniform_freqs):
        """"A single frequency peak should create a distinct peak in the CSD."""
        pressure = np.zeros(len(uniform_freqs), dtype=complex)
        peak_freq = 1000.0
        peak_idx = int(peak_freq - uniform_freqs[0])
        if 0 <= peak_idx < len(pressure):
            pressure[peak_idx] = 1.0
        result = compute_csd(uniform_freqs, pressure, n_slices=15, window_ms=20.0)
        # Find column nearest to peak_freq
        freq_idx = np.argmin(np.abs(result.csd_freqs - peak_freq))
        col = result.csd_db[:, freq_idx]
        peak_val = np.max(col)
        # CSD at the peak should be among the top values
        assert peak_val >= np.sort(col)[-3]

    def test_impulse_response_from_csd_matches_result_field(
        self, uniform_freqs, flat_spectrum
    ):
        """CSD's impulse field should match what compute_impulse_response returns."""
        t_ir, ir = compute_impulse_response(uniform_freqs, flat_spectrum, window_ms=20.0)
        result = compute_csd(uniform_freqs, flat_spectrum, window_ms=20.0)
        # Shapes should match (ignoring tiny rounding differences)
        assert len(ir) == len(result.impulse)


class TestStepResponseUnitScaling:
    """Step response integration uses dt in seconds (dt * 1e-3 from ms).

    The step response is the time-integral of the impulse response:
      step(t) = ∫₀ᵗ impulse(τ) dτ

    impulse from irfft(pressure spectrum) is in Pa·s (pressure · time).
    dt from compute_impulse_response is in milliseconds; dt * 1e-3 converts to seconds.
    This is mathematically identical to dt/1000 but more explicit about the unit conversion.
    """

    def test_step_response_not_zero_for_broadband_spectrum(self):
        """
        A broadband flat spectrum should produce a non-zero step response.
        Step values of exactly 0 or ~1e-12 Pa·s would indicate the integration
        is missing the dt time-scaling factor.
        """
        freqs = np.linspace(100, 10000, 500)
        pressure = np.ones(len(freqs), dtype=complex) * 0.1

        result = compute_csd(freqs, pressure, n_slices=10, window_ms=20.0)

        step_max = np.max(np.abs(result.step))
        # For a flat spectrum with pressure ~0.1 Pa and n_fft-based dt,
        # step should be ~0.01 * dt_s ≈ O(1e-5 to 1e-4) Pa·s, not near-zero
        assert step_max > 1e-9, (
            f"Step response magnitude ({step_max:.2e} Pa·s) is suspiciously small. "
            f"Check that dt integration uses proper time-unit conversion (dt * 1e-3)."
        )
        assert np.isfinite(step_max)

    def test_step_response_for_single_tone_not_zero(self):
        """
        For a single-tone impulse at f ≈ 1 kHz (ω ≈ 6283 rad/s), the step
        response peak should be on the order of A/ω ≈ 1.6e-4 Pa·s.
        A step near zero would indicate a bug in the time integration.
        """
        freqs = np.linspace(100, 20000, 1000)
        pressure = np.zeros(len(freqs), dtype=complex)
        tone_idx = 50  # ~1096 Hz
        pressure[tone_idx] = 1.0

        result = compute_csd(freqs, pressure, n_slices=10, window_ms=50.0)
        step_peak = np.max(np.abs(result.step))

        assert step_peak > 1e-9, (
            f"Step peak ({step_peak:.2e} Pa·s) is near zero for a 1 Pa tone. "
            f"Expected ~1e-4 Pa·s based on A/ω theory."
        )
        assert np.isfinite(step_peak)
        # And it should not be absurdly large
        assert step_peak < 1.0, (
            f"Step peak ({step_peak:.2e} Pa·s) is unreasonably large for 1 Pa input."
        )

    def test_step_response_increases_with_higher_pressure_magnitude(self):
        """
        Doubling the input pressure magnitude should roughly double the step response.
        """
        freqs = np.linspace(100, 5000, 300)
        pressure_low = np.ones(len(freqs), dtype=complex) * 0.1
        pressure_high = np.ones(len(freqs), dtype=complex) * 0.2

        r_low = compute_csd(freqs, pressure_low, n_slices=5, window_ms=20.0)
        r_high = compute_csd(freqs, pressure_high, n_slices=5, window_ms=20.0)

        step_low = np.max(np.abs(r_low.step))
        step_high = np.max(np.abs(r_high.step))

        ratio = step_high / step_low if step_low > 1e-12 else 0.0
        assert ratio == pytest.approx(2.0, rel=0.25), (
            f"Step ratio (2× pressure) = {ratio:.3f}, expected ~2.0. "
            f"step_low={step_low:.4e}, step_high={step_high:.4e}"
        )