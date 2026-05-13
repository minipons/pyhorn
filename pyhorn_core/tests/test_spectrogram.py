"""
Tests for pyhorn_core.solver.spectrogram.
"""

import numpy as np
import pytest
from pyhorn_core.solver.spectrogram import (
    compute_impulse_response,
    spectrogram_data,
    plot_spectrogram,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_response(freqs: np.ndarray) -> tuple:
    """Return synthetic SPL (dB) and phase (rad) for a sum of resonances.

    Uses scipy.signal.minimum_phase to build a causal minimum-phase impulse
    whose magnitude spectrum has peaks at 200 Hz and 800 Hz.  This produces
    a genuinely causal impulse response through compute_impulse_response(),
    unlike zero-phase Gaussians which create a symmetric (non-causal) impulse.
    """
    from scipy.signal import minimum_phase

    n = 8192
    fs = 44100.0
    fft_freqs = np.fft.rfftfreq(n, 1 / fs)

    # Build magnitude with two narrow resonance peaks
    H_mag = np.ones(len(fft_freqs)) * 1e-8  # very low floor
    H_mag += 1.0 * np.exp(-((fft_freqs - 200.0) ** 2) / (2 * 10.0**2))  # 200 Hz peak
    H_mag += 1.0 * np.exp(-((fft_freqs - 800.0) ** 2) / (2 * 20.0**2))  # 800 Hz peak
    H_mag[0] = 1e-12  # suppress DC to avoid log(0) issues

    # Minimum phase: produces a causal impulse with peak at t=0
    H_min = minimum_phase(H_mag**2, method="hilbert")

    # Shift impulse to start at t > 0 (purely causal)
    shift = 500  # samples
    h = np.zeros_like(H_min)
    h[shift:] = H_min[:-shift]
    h = h / np.max(np.abs(h)) * 0.05

    # Convert to frequency domain on the native FFT grid (zero-pad to n points)
    H = np.fft.rfft(h, n=n)

    # Interpolate to the requested frequency grid
    spl = np.interp(freqs, fft_freqs, 20.0 * np.log10(np.abs(H) + 1e-12))
    phase = np.interp(freqs, fft_freqs, np.angle(H))
    return spl, phase


def _causal_impulse(freqs, spl, phase):
    """Return a causal (t>=0) impulse response from a frequency response."""
    t_ms, h = compute_impulse_response(freqs, spl, phase)
    peak_idx = np.argmax(np.abs(h))
    t_causal = t_ms[peak_idx:] - t_ms[peak_idx]
    h_causal = h[peak_idx:]
    return t_causal, h_causal


class TestImpulseResponse:
    def test_returns_real_impulse(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_ms, h = compute_impulse_response(freqs, spl, phase, fs=44100.0)
        assert t_ms.shape == h.shape
        assert np.isrealobj(h)

    def test_impulse_peak_near_zero(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_ms, h = compute_impulse_response(freqs, spl, phase, fs=44100.0)
        peak_idx = np.argmax(np.abs(h))
        assert abs(t_ms[peak_idx]) < 5.0, "Impulse peak should be near t=0 ms"

    def test_rejects_single_frequency_point(self):
        freqs = np.array([100.0])
        with pytest.raises(ValueError, match="at least"):
            compute_impulse_response(freqs, np.array([80.0]), np.array([0.0]))


class TestSpectrogramData:
    def test_output_shapes_consistent(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        time_bins, freq_bins, stft_db = spectrogram_data(t_causal, h_causal, window_ms=5.0)
        assert time_bins.ndim == 1
        assert freq_bins.ndim == 1
        assert stft_db.ndim == 2
        assert stft_db.shape[0] == len(time_bins)  # (n_windows, n_freqs)
        assert stft_db.shape[1] == len(freq_bins)

    def test_time_axis_monotonic(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        time_bins, _, _ = spectrogram_data(t_causal, h_causal, window_ms=5.0)
        assert np.all(np.diff(time_bins) > 0)

    def test_db_values_finite(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        _, _, stft_db = spectrogram_data(t_causal, h_causal, window_ms=5.0)
        assert np.all(np.isfinite(stft_db))

    def test_db_values_bounded(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        _, _, stft_db = spectrogram_data(t_causal, h_causal, window_ms=5.0)
        assert np.max(stft_db) < 200
        assert np.max(stft_db) > -200

    def test_resonance_peaks_visible(self):
        """High-SPL (resonance) frequencies should show as bright bands."""
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        time_bins, freq_bins, stft_db = spectrogram_data(t_causal, h_causal, window_ms=5.0)

        f1_idx = np.argmin(np.abs(freq_bins - 200.0))
        f2_idx = np.argmin(np.abs(freq_bins - 800.0))
        early_mask = time_bins <= 10.0  # first 10 ms

        # stft_db is in dB re: peak; the global maximum should be ~0 dB by
        # construction (normalised by the peak magnitude).  Use 0 dB as the
        # reference for the visibility threshold.
        peak_200 = np.max(stft_db[early_mask, f1_idx])
        peak_800 = np.max(stft_db[early_mask, f2_idx])

        # Both resonances should appear as bright bands (within 40 dB of 0 dB)
        assert peak_200 > -40, f"200 Hz resonance should appear as bright band (got {peak_200:.1f} dB)"
        assert peak_800 > -40, f"800 Hz resonance should appear as bright band (got {peak_800:.1f} dB)"

    def test_more_overlap_gives_more_windows(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        _, _, spec_low = spectrogram_data(t_causal, h_causal, window_ms=5.0, overlap=0.1)
        _, _, spec_high = spectrogram_data(t_causal, h_causal, window_ms=5.0, overlap=0.9)
        # stft_db is (n_windows, n_freqs): shape[0] is n_windows
        assert spec_high.shape[0] > spec_low.shape[0]

    def test_smaller_window_gives_more_windows(self):
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        _, _, spec_20 = spectrogram_data(t_causal, h_causal, window_ms=2.0)
        _, _, spec_100 = spectrogram_data(t_causal, h_causal, window_ms=10.0)
        # stft_db is (n_windows, n_freqs): shape[0] is n_windows
        assert spec_20.shape[0] > spec_100.shape[0]

    def test_raises_on_invalid_overlap(self):
        freqs = np.linspace(20, 5000, 200)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        with pytest.raises(ValueError, match="overlap"):
            spectrogram_data(t_causal, h_causal, overlap=-0.1)
        with pytest.raises(ValueError, match="overlap"):
            spectrogram_data(t_causal, h_causal, overlap=1.0)


class TestSpectrogramPlot:
    def test_plot_returns_figure_and_axes(self):
        import matplotlib.pyplot as plt
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        time_bins, freq_bins, stft_db = spectrogram_data(t_causal, h_causal, window_ms=5.0)
        fig_in, ax_in = plt.subplots()
        fig_result, ax_result = plot_spectrogram(time_bins, freq_bins, stft_db, ax=ax_in)
        assert ax_result is ax_in
        assert fig_result is fig_in
        plt.close(fig_result)

    def test_plot_with_custom_figsize(self):
        import matplotlib.pyplot as plt
        freqs = np.linspace(20, 5000, 500)
        spl, phase = _make_response(freqs)
        t_causal, h_causal = _causal_impulse(freqs, spl, phase)
        time_bins, freq_bins, stft_db = spectrogram_data(t_causal, h_causal, window_ms=10.0)
        fig, ax = plt.subplots()
        fig_result, ax_result = plot_spectrogram(time_bins, freq_bins, stft_db, figsize=(8, 4), ax=ax)
        assert ax_result is ax
        assert fig_result is fig
        plt.close(fig_result)
