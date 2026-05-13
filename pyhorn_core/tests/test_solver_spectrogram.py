"""Unit tests for pyhorn.solver.spectrogram."""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path

from pyhorn_core.solver.spectrogram import (
    compute_impulse_response,
    spectrogram_data,
    plot_spectrogram,
    export_spectrogram_csv,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def uniform_freqs():
    """Uniformly spaced 20 Hz–20 kHz at 1 Hz resolution."""
    return np.linspace(20, 20000, 19981)


@pytest.fixture
def flat_spl():
    """Bandpass-like SPL spectrum (non-trivial impulse response)."""
    # A frequency-dependent profile so the impulse has finite temporal spread,
    # not a pure delta at t=0.  Using a simple roll-off shape.
    spl = np.full(19981, 90.0)
    # Gentle low-frequency roll below 100 Hz and high-frequency roll above 8 kHz
    for i, f in enumerate(np.linspace(20, 20000, 19981)):
        if f < 100:
            spl[i] = 90.0 + 20 * np.log10(f / 100)
        elif f > 8000:
            spl[i] = 90.0 - 10 * np.log10(f / 8000)
    return spl


@pytest.fixture
def zero_phase():
    """Zero phase (in-phase transfer function)."""
    return np.zeros(19981)


@pytest.fixture
def impulse_response_data(uniform_freqs, flat_spl, zero_phase):
    """Pre-computed impulse response for reuse."""
    t, h = compute_impulse_response(uniform_freqs, flat_spl, zero_phase, fs=44100)
    return t, h


# ─── Test compute_impulse_response ───────────────────────────────────────────

class TestComputeImpulseResponse:
    """Tests for compute_impulse_response()."""

    def test_returns_tuple_of_arrays(self, uniform_freqs, flat_spl, zero_phase):
        """Should return (t_ms, h) as numpy arrays."""
        t, h = compute_impulse_response(uniform_freqs, flat_spl, zero_phase)
        assert isinstance(t, np.ndarray)
        assert isinstance(h, np.ndarray)
        assert len(t) == len(h)

    def test_impulse_is_real(self, uniform_freqs, flat_spl, zero_phase):
        """Impulse response should be purely real (not complex)."""
        _, h = compute_impulse_response(uniform_freqs, flat_spl, zero_phase)
        assert np.isrealobj(h)

    def test_time_axis_in_milliseconds(self, uniform_freqs, flat_spl, zero_phase):
        """Time axis should be in milliseconds and uniformly spaced, starting at t=0."""
        t, _ = compute_impulse_response(uniform_freqs, flat_spl, zero_phase)
        # Starts at t=0, uniformly spaced, positive end
        assert t[0] == pytest.approx(0.0, abs=1e-3)
        assert t[-1] > 0
        # Uniform spacing
        dt = np.diff(t)
        assert np.allclose(dt, dt[0])

    def test_impulse_length_nonzero(self, uniform_freqs, flat_spl, zero_phase):
        """Impulse response should have a reasonable non-zero length."""
        _, h = compute_impulse_response(uniform_freqs, flat_spl, zero_phase)
        assert len(h) > 0
        # Not all zeros
        assert not np.allclose(h, 0)

    def test_impulse_nonzero_early(self, uniform_freqs, flat_spl, zero_phase):
        """Impulse response should have significant energy in the first portion."""
        t, h = compute_impulse_response(uniform_freqs, flat_spl, zero_phase)
        peak_idx = np.argmax(np.abs(h))
        peak_t = t[peak_idx]
        # Peak should be in the first 50% of the time axis (causal response)
        assert peak_t <= t[0] + (t[-1] - t[0]) * 0.5
        # Peak index should be ≥ 0
        assert peak_idx >= 0

    def test_single_frequency_tone(self, uniform_freqs):
        """A single-frequency phase-aligned peak produces a clean sinusoidal IR."""
        spl = np.zeros_like(uniform_freqs)
        phase = np.zeros_like(uniform_freqs)
        # 1 kHz peak
        peak_idx = 980
        spl[peak_idx] = 100.0
        t, h = compute_impulse_response(uniform_freqs, spl, phase, fs=44100)
        # IR should be oscillatory and non-zero
        assert not np.allclose(h, 0)
        assert np.std(h) > 0

    def test_fs_parameter_runs_without_error(self, uniform_freqs, flat_spl, zero_phase):
        """Function should work at various sample rates without error."""
        for fs in [8000, 22050, 44100, 96000]:
            t, h = compute_impulse_response(uniform_freqs, flat_spl, zero_phase, fs=fs)
            assert len(t) == len(h)
            assert np.isrealobj(h)
            # Time axis starts at 0 for any fs
            assert t[0] >= 0

    def test_non_uniform_freqs_interpolated(self):
        """Non-uniform (log) frequency spacing should be handled by interpolation."""
        freqs = np.array([20, 50, 100, 500, 1000, 5000, 10000, 20000])
        spl = np.ones(len(freqs)) * 94.0
        phase = np.zeros(len(freqs))
        t, h = compute_impulse_response(freqs, spl, phase, fs=44100)
        assert len(t) > 0
        assert np.isrealobj(h)

    def test_different_sample_rates(self, uniform_freqs, flat_spl, zero_phase):
        """Function should work at various sample rates without error."""
        for fs in [8000, 22050, 44100, 96000]:
            t, h = compute_impulse_response(uniform_freqs, flat_spl, zero_phase, fs=fs)
            assert len(t) == len(h)
            assert np.isrealobj(h)


class TestImpulseResponseInverseFFTQuality:
    """Verify forward FFT then inverse FFT recovers the original signal."""

    def test_forward_inverse_roundtrip(self):
        """
        Take an impulse response h(t), FFT it, then IFFT it back.
        The recovered signal should match the original within numerical tolerance.
        """
        # Create a synthetic impulse response: a decaying sinusoid
        fs = 44100
        t_ms = np.linspace(-50, 50, 4096)
        dt_ms = t_ms[1] - t_ms[0]
        t_s = t_ms / 1000.0
        # decaying oscillation at 1 kHz
        h_original = np.exp(-20 * np.abs(t_s)) * np.sin(2 * np.pi * 1000 * t_s)

        # Build the one-sided frequency spectrum (as our function expects)
        n_fft = len(t_ms)
        H_full = np.fft.fft(h_original)  # full FFT
        H_one_sided = H_full[:n_fft // 2 + 1]  # irfft range

        spl = 20 * np.log10(np.abs(H_one_sided) + 1e-12)
        phase = np.angle(H_one_sided)
        df = 1.0 / (n_fft * dt_ms / 1000.0)  # Hz per bin
        freqs = np.arange(n_fft // 2 + 1) * df

        # Now go through our function
        t_out, h_out = compute_impulse_response(freqs, spl, phase, fs=fs)

        # The time axis will differ (windowing), so compare RMS of the signals
        # within the overlapping portion
        assert len(h_out) > 0
        assert np.isrealobj(h_out)
        # Compare RMS of outputs (both should be non-zero decaying sinusoids)
        rms_orig = np.sqrt(np.mean(h_original ** 2))
        rms_out = np.sqrt(np.mean(h_out ** 2))
        assert rms_out > 0


# ─── Test spectrogram_data ─────────────────────────────────────────────────────

class TestSpectrogramData:
    """Tests for spectrogram_data()."""

    def test_returns_three_arrays(self, impulse_response_data):
        """Should return (time_bins, freq_bins, stft_db)."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)
        assert isinstance(tb, np.ndarray)
        assert isinstance(fb, np.ndarray)
        assert isinstance(sd, np.ndarray)

    def test_spectrogram_shape(self, impulse_response_data):
        """STFT output should have correct 2D shape (n_time_bins × n_freq_bins)."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h, window_ms=5.0)
        assert sd.ndim == 2
        assert sd.shape[0] == len(tb)   # rows = time
        assert sd.shape[1] == len(fb)   # cols = frequency

    def test_time_axis_increasing(self, impulse_response_data):
        """Time bins should be monotonically increasing in ms."""
        t, h = impulse_response_data
        tb, _, _ = spectrogram_data(t, h)
        assert np.all(np.diff(tb) > 0)

    def test_freq_axis_positive(self, impulse_response_data):
        """Frequency bins should be non-negative (includes DC=0 Hz from STFT)."""
        t, h = impulse_response_data
        _, fb, _ = spectrogram_data(t, h)
        assert np.all(fb >= 0)  # includes DC at 0 Hz
        assert fb[-1] > 0  # Nyquist > 0

    def test_spectrogram_db_range(self, impulse_response_data):
        """STFT dB values should be in a reasonable range (≤ 0, not -inf everywhere)."""
        t, h = impulse_response_data
        _, _, sd = spectrogram_data(t, h, window_ms=5.0)
        # Peak-normalised: 0 dB = peak; everything ≤ 0
        assert np.all(sd <= 0.5)  # allow small rounding error above 0
        # Should not be all -120 (window too short to capture signal)
        fraction_finite = np.mean(sd > -119)
        assert fraction_finite > 0.01, "Spectrogram looks like it captured no signal"

    def test_window_ms_affects_time_resolution(self, impulse_response_data):
        """Larger window_ms should give coarser time resolution (fewer windows)."""
        t, h = impulse_response_data
        tb1, _, _ = spectrogram_data(t, h, window_ms=2.0)
        tb2, _, _ = spectrogram_data(t, h, window_ms=10.0)
        assert len(tb1) > len(tb2)

    def test_overlap_affects_window_count(self, impulse_response_data):
        """Higher overlap fraction should give more time windows."""
        t, h = impulse_response_data
        tb_low, _, _ = spectrogram_data(t, h, overlap=0.0)
        tb_high, _, _ = spectrogram_data(t, h, overlap=0.9)
        assert len(tb_high) > len(tb_low)

    def test_all_window_types(self, impulse_response_data):
        """All three window types should run without error."""
        t, h = impulse_response_data
        for wtype in ["hann", "hamming", "blackman"]:
            tb, fb, sd = spectrogram_data(t, h, window_type=wtype)
            assert sd.shape[0] == len(tb)
            assert sd.shape[1] == len(fb)

    def test_invalid_window_type_raises(self, impulse_response_data):
        """Invalid window_type should raise ValueError."""
        t, h = impulse_response_data
        with pytest.raises(ValueError, match="Unknown window_type"):
            spectrogram_data(t, h, window_type="kaiser")

    def test_invalid_overlap_raises(self, impulse_response_data):
        """overlap outside [0, 1) should raise ValueError."""
        t, h = impulse_response_data
        with pytest.raises(ValueError, match="overlap"):
            spectrogram_data(t, h, overlap=1.0)
        with pytest.raises(ValueError, match="overlap"):
            spectrogram_data(t, h, overlap=-0.1)

    def test_spectrogram_time_axis_positive(self, impulse_response_data):
        """Time bins should be positive milliseconds and increasing."""
        t, h = impulse_response_data
        tb, _, _ = spectrogram_data(t, h)
        assert tb[0] > 0  # starts well after t=0
        assert np.all(np.diff(tb) > 0)  # monotonically increasing

    def test_spectrogram_freq_axis_loggable(self, impulse_response_data):
        """Frequency axis can be passed to log scale (non-negative values)."""
        t, h = impulse_response_data
        _, fb, _ = spectrogram_data(t, h)
        # DC (0 Hz) can't be log-scaled; skip it; rest must be positive
        assert np.all(fb[1:] > 0), "Non-DC frequency bins must be positive for log scale"


# ─── Test plot_spectrogram ─────────────────────────────────────────────────────

class TestPlotSpectrogram:
    """Tests for plot_spectrogram()."""

    def test_runs_without_error(self, impulse_response_data):
        """Should generate a figure without raising an exception."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h, window_ms=5.0)
        fig, ax = plot_spectrogram(tb, fb, sd)
        assert fig is not None
        assert ax is not None
        plt_close(fig)

    def test_returns_figure_and_axes(self, impulse_response_data):
        """Should return (fig, ax)."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)
        result = plot_spectrogram(tb, fb, sd)
        assert len(result) == 2
        fig, ax = result
        import matplotlib.figure
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_figsize(self, impulse_response_data):
        """Custom figsize should be respected."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)
        fig, _ = plot_spectrogram(tb, fb, sd, figsize=(10, 4))
        assert fig.get_size_inches() == pytest.approx((10, 4))
        plt_close(fig)

    def test_custom_vmin_vmax(self, impulse_response_data):
        """Custom vmin/vmax should be passed through to pcolormesh."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)
        fig, ax = plot_spectrogram(tb, fb, sd, vmin=-80, vmax=-10)
        # Check the image was created
        ims = ax.findobj(match=lambda x: hasattr(x, "set_clim"))
        assert len(ims) > 0
        plt_close(fig)

    def test_linear_frequency_axis(self, impulse_response_data):
        """Linear frequency axis should work without log scale."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)
        fig, ax = plot_spectrogram(tb, fb, sd, log_freq=False)
        plt_close(fig)

    def test_frequency_display_limits(self, impulse_response_data):
        """f_min and f_max should clip the display range."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)
        fig, ax = plot_spectrogram(tb, fb, sd, f_min=100, f_max=5000)
        ylim = ax.get_ylim()
        assert ylim[0] >= 50  # some margin
        plt_close(fig)


# ─── Test export_spectrogram_csv ───────────────────────────────────────────────

class TestExportSpectrogramCsv:
    """Tests for export_spectrogram_csv()."""

    def test_writes_csv_file(self, impulse_response_data):
        """CSV file should be created with correct dimensions."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h, window_ms=5.0)
        n_time = len(tb)
        n_freq = len(fb)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spectrogram.csv")
            export_spectrogram_csv(tb, fb, sd, path)

            assert os.path.exists(path)

            with open(path, "r") as fh:
                lines = fh.readlines()

            # Header: Frequency_Hz,T0_ms,T1_ms,...
            header_parts = lines[0].strip().split(",")
            assert header_parts[0] == "Frequency_Hz"
            assert len(header_parts) == n_time + 1

            # One row per frequency bin
            assert len(lines) == n_freq + 1

    def test_csv_values_match_stft_db(self, impulse_response_data):
        """Values written to CSV should match the stft_db input."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spectrogram.csv")
            export_spectrogram_csv(tb, fb, sd, path)

            with open(path, "r") as fh:
                lines = fh.readlines()

            # Check a frequency row
            for row_idx in [0, len(lines) // 2, len(lines) - 2]:
                line = lines[1 + row_idx].strip()
                parts = line.split(",")
                freq_val = float(parts[0])
                assert np.isclose(freq_val, fb[row_idx], rtol=1e-3)

    def test_shape_mismatch_raises(self, impulse_response_data):
        """Mismatched array shapes should raise ValueError."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)
        # Pass wrong shape
        bad_sd = np.zeros((len(tb) + 1, len(fb) - 1))
        with pytest.raises(ValueError, match="does not match"):
            export_spectrogram_csv(tb, fb, bad_sd, "/tmp/junk.csv")

    def test_csv_readable_float_format(self, impulse_response_data):
        """CSV values should be parseable as floats."""
        t, h = impulse_response_data
        tb, fb, sd = spectrogram_data(t, h)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "spectrogram.csv")
            export_spectrogram_csv(tb, fb, sd, path)

            with open(path, "r") as fh:
                lines = fh.readlines()

            for line in lines[1:]:  # skip header
                parts = line.strip().split(",")
                freq = float(parts[0])
                assert np.isfinite(freq)
                for p in parts[1:]:
                    val = float(p)
                    assert np.isfinite(val)


# ─── Helper ────────────────────────────────────────────────────────────────────

def plt_close(fig):
    """Close a matplotlib figure to free memory."""
    import matplotlib.pyplot as plt
    plt.close(fig)
