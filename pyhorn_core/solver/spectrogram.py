"""
Spectrogram / Short-Time Fourier Transform (STFT) analysis for horn responses.

Displays spectral intensity of the impulse response or normalised amplitude in dB
as a function of frequency vs. time (ms), matching Hornresp's spectrogram display
(Hornresp page 97).

The pipeline is:
  1. Reconstruct complex transfer function H(jω) from SPL + phase
  2. IFFT → impulse response h(t)
  3. Apply STFT (sliding window + FFT) → 2-D time-frequency energy map
  4. Convert to dB SPL
"""

import matplotlib

matplotlib.use("Agg")  # Headless-safe: must be set before pyplot import

import numpy as np
from scipy import signal as scipy_signal
from typing import Tuple, Optional
import matplotlib.pyplot as plt


# ─── 1. Impulse response from SPL + phase ─────────────────────────────────────

def compute_impulse_response(
    frequencies: np.ndarray,
    spl: np.ndarray,
    phase: np.ndarray,
    fs: float = 44100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the impulse response h(t) from a horn frequency response.

    Reconstructs the complex transfer function:
        H(jω) = 10^(spl/20) × e^(j × phase)

    then applies the inverse FFT to obtain the time-domain impulse response.

    Parameters
    ----------
    frequencies : array of shape (n_freq,) — frequency grid in Hz.
                  Must be uniformly spaced; will be interpolated if not.
    spl         : array of shape (n_freq,) — SPL values in dB SPL.
    phase       : array of shape (n_freq,) — phase values in radians.
    fs          : sample rate in Hz for the impulse response (default 44100).

    Returns
    -------
    t_ms : array of shape (n_samples,) — time axis in milliseconds.
           The impulse is centred at t=0 (zero-phase window applied).
    h    : array of shape (n_samples,) — impulse response amplitude (Pa).
           Purely real (not complex).
    """
    frequencies = np.asarray(frequencies)
    spl = np.asarray(spl)
    phase = np.asarray(phase)

    if len(frequencies) < 2:
        raise ValueError("Need at least 2 frequency points.")

    f_min = float(frequencies[0])
    f_max = float(frequencies[-1])

    # ── Interpolate to a uniform frequency grid ───────────────────────────────
    df_input = (f_max - f_min) / max(len(frequencies) - 1, 1)
    # n_fft: large enough to represent the impulse at sample rate fs,
    # and zero-padded to a power of 2 for efficient FFT.
    n_fft = int(2 ** np.ceil(np.log2(
        max(int(np.ceil(fs / df_input)), len(frequencies) * 2)
    )))

    freqs_uniform = np.linspace(f_min, f_max, n_fft)
    mag_interp = np.interp(freqs_uniform, frequencies, spl)
    phase_interp = np.interp(
        freqs_uniform,
        frequencies,
        np.unwrap(phase),
    )

    # Reconstruct H(jω) = 10^(spl/20) × e^(jφ)
    H = 10 ** (mag_interp / 20.0) * np.exp(1j * phase_interp)

    # Build one-sided spectrum (DC to Nyquist) for irfft
    full_spectrum = np.zeros(n_fft // 2 + 1, dtype=complex)
    full_spectrum[:] = H[: n_fft // 2 + 1]

    # Inverse FFT → impulse response (real-valued, causal: starts at t=0)
    h = np.fft.irfft(full_spectrum, n=n_fft)

    # Time axis: starts at 0, ends at (n_fft-1)/fs seconds
    dt_s = 1.0 / fs
    t = np.arange(n_fft) * dt_s * 1000.0  # ms

    return t, h


# ─── 2. STFT / spectrogram data ───────────────────────────────────────────────

def spectrogram_data(
    t: np.ndarray,
    h: np.ndarray,
    window_ms: float = 5.0,
    overlap: float = 0.5,
    window_type: str = "hann",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT spectrogram data from an impulse response.

    Parameters
    ----------
    t           : time axis in milliseconds (uniformly spaced, starting at ≥0).
    h           : impulse response amplitude array (real-valued).
    window_ms   : analysis window length in milliseconds (default 5.0).
                  Smaller → better time resolution, worse frequency resolution.
    overlap     : fraction of overlap between adjacent windows, in [0, 1)
                  (default 0.5 = 50% overlap).
    window_type : window function — 'hann' | 'hamming' | 'blackman'
                  (default 'hann').

    Returns
    -------
    time_bins   : array of shape (n_windows,) — centre time of each window (ms).
                  Monotonically increasing.
    freq_bins   : array of shape (n_freq_bins,) — frequency axis of the STFT (Hz).
                  Always positive, increasing.
    stft_db     : array of shape (n_windows, n_freq_bins) — magnitude in dB re: peak.
                  0 dB = peak of the spectrogram.
    """
    if overlap < 0 or overlap >= 1:
        raise ValueError("overlap must be in [0, 1).")

    h = np.asarray(h, dtype=np.float64)
    t = np.asarray(t)

    if len(h) < 2:
        raise ValueError("Impulse response must have at least 2 samples.")

    # dt in milliseconds
    dt_ms = (t[-1] - t[0]) / max(len(t) - 1, 1)
    if dt_ms <= 0:
        raise ValueError("Time axis must be uniformly increasing.")
    dt_s = dt_ms / 1000.0

    # Window length in samples
    window_samples = max(1, int(round(window_ms / dt_ms)))
    window_samples = min(window_samples, len(h))

    # hop
    noverlap = int(round(window_samples * overlap))
    noverlap = min(noverlap, window_samples - 1)

    # Window function
    if window_type == "hann":
        window_fn = np.hanning(window_samples)
    elif window_type == "hamming":
        window_fn = np.hamming(window_samples)
    elif window_type == "blackman":
        window_fn = np.blackman(window_samples)
    else:
        raise ValueError(f"Unknown window_type '{window_type}'. "
                         "Expected 'hann' | 'hamming' | 'blackman'.")

    # scipy STFT: returns (n_freq_bins, n_windows) — we transpose to (n_windows, n_freq_bins)
    freqs, _times, Zxx = scipy_signal.stft(
        h,
        fs=1.0 / dt_s,
        window=window_fn,
        nperseg=window_samples,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    # Compute time bin centres: scipy returns frame start times in seconds
    # time_bins = start + (window_centroid) * dt_s in ms
    hop = window_samples - noverlap
    frame_centres_s = (np.arange(Zxx.shape[1]) * hop + window_samples / 2) * dt_s
    time_bins = frame_centres_s * 1000.0  # ms

    # magnitude in dB re: peak; transpose to (n_windows, n_freq_bins)
    magnitude = np.abs(Zxx).T
    peak = np.max(magnitude)
    if peak <= 0:
        stft_db = np.full((Zxx.shape[1], Zxx.shape[0]), -120.0)
    else:
        stft_db = 20.0 * np.log10(magnitude / peak + 1e-12)

    return time_bins, freqs, stft_db


# ─── 3. Plotting ──────────────────────────────────────────────────────────────

_FREQ_TICKS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]


def _fmt_hz(val, pos):
    v = int(val)
    if v >= 1000:
        return f"{v // 1000}k"
    return str(v)


def plot_spectrogram(
    time_bins: np.ndarray,
    freq_bins: np.ndarray,
    stft_db: np.ndarray,
    figsize: Tuple[float, float] = (12, 6),
    vmin: float = -60.0,
    vmax: float = 0.0,
    cmap: str = "magma",
    log_freq: bool = True,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    window_ms: Optional[float] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Render a spectrogram as a matplotlib pcolormesh panel.

    Parameters
    ----------
    time_bins  : centre time of each STFT window (ms), shape (n_windows,).
    freq_bins  : frequency bins of the STFT (Hz), shape (n_freq_bins,).
    stft_db    : 2-D magnitude in dB re: peak, shape (n_windows, n_freq_bins).
    figsize    : figure size in inches (default (12, 6)).
    vmin       : minimum dB for colour scale (default -60).
    vmax       : maximum dB for colour scale (default 0).
    cmap       : colormap name (default 'magma').
    log_freq   : use logarithmic frequency axis (default True).
    f_min      : lower display frequency limit in Hz (default: auto from freq_bins).
    f_max      : upper display frequency limit in Hz (default: auto from freq_bins).
    ax         : matplotlib Axes; if None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure
    ax  : matplotlib Axes with the spectrogram
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.get_figure()

    # Frequency display range
    if f_min is None:
        f_min = float(freq_bins[0])
    if f_max is None:
        f_max = float(freq_bins[-1])
    # For log scale: ensure f_min > 0 (can't log-scale DC)
    if log_freq and f_min <= 0:
        f_min = max(f_min, 1.0)  # at least 1 Hz

    # Clip to data range
    freq_display_mask = (freq_bins >= f_min) & (freq_bins <= f_max)
    freq_display = freq_bins[freq_display_mask]
    stft_display = stft_db[:, freq_display_mask]

    # Time range
    t_min = float(time_bins[0])
    t_max = float(time_bins[-1])

    # Compute bin edges for pcolormesh
    df_arr = np.diff(freq_display)
    if len(df_arr) == 0:
        # Single frequency bin: estimate width from bin spacing
        df_arr = np.array([freq_bins[1] - freq_bins[0]] if len(freq_bins) > 1 else [1.0])
    else:
        df_arr = np.concatenate([[df_arr[0]], df_arr])
    f_edges = np.concatenate([freq_display - df_arr / 2,
                               [freq_display[-1] + df_arr[-1] / 2]])
    f_edges = np.clip(f_edges, f_min, f_max)

    dt_arr = np.diff(time_bins)
    if len(dt_arr) == 0:
        # Single-window STFT: estimate bin width from frequency resolution / sample rate
        dt_arr = np.array([1.0])  # 1 ms placeholder
    else:
        dt_arr = np.concatenate([[dt_arr[0]], dt_arr])
    t_edges = np.concatenate([time_bins - dt_arr / 2,
                               [time_bins[-1] + dt_arr[-1] / 2]])
    t_edges = np.clip(t_edges, t_min, t_max)

    mesh = ax.pcolormesh(
        t_edges,
        f_edges,
        stft_display.T,  # transpose: (n_freq, n_time)
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="flat",
    )

    # Colour bar
    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label("dB (re: peak)", fontsize=9)

    # Frequency axis
    if log_freq:
        ax.set_yscale("log")
    ax.set_ylim(f_min, f_max)
    ax.set_xlim(t_min, t_max)

    if log_freq:
        from matplotlib.ticker import FixedLocator, FuncFormatter
        ax.yaxis.set_major_locator(FixedLocator(_FREQ_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_hz))

    ax.set_xlabel("Time (ms)", fontsize=9)
    ax.set_ylabel("Frequency (Hz)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(title if title is not None else "Horn Spectrogram", fontsize=10, fontweight="medium")

    return fig, ax


# ─── 4. CSV export ────────────────────────────────────────────────────────────

def compute_spectrogram(
    frequencies: np.ndarray,
    pressure: np.ndarray,
    window_ms: float = 50.0,
    overlap: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a spectrogram from a complex pressure frequency response.

    This is a convenience wrapper that combines compute_impulse_response and
    spectrogram_data into a single call.

    Parameters
    ----------
    frequencies : array of shape (n_freq,) — log-spaced frequency grid (Hz).
    pressure   : complex array of shape (n_freq,) — complex pressure at 1 m (Pa).
    window_ms  : STFT window duration in milliseconds (default 50 ms).
    overlap    : fraction of overlap between adjacent windows (default 0.5).

    Returns
    -------
    time_ms        : array of shape (n_windows,) — centre time of each window (ms).
    freq_bins_hz   : array of shape (n_freq_bins,) — frequency axis of the STFT (Hz).
    spectrogram_db : array of shape (n_windows, n_freq_bins) — magnitude in dB re: peak.
    """
    # Extract SPL (dB) and phase (rad) from complex pressure
    spl = 20.0 * np.log10(np.abs(pressure) + 1e-12)
    phase = np.angle(pressure)

    # Compute impulse response
    t, h = compute_impulse_response(frequencies, spl, phase, fs=44100.0)

    # Compute spectrogram
    return spectrogram_data(t, h, window_ms=window_ms, overlap=overlap)


def export_spectrogram_csv(
    time_bins: np.ndarray,
    freq_bins: np.ndarray,
    stft_db: np.ndarray,
    path: str,
) -> None:
    """
    Export spectrogram data to a CSV file.

    The CSV has columns:
        Frequency_Hz, T0_ms, T1_ms, ..., TN_ms

    Each row corresponds to a frequency bin; the first column is the
    centre frequency in Hz, followed by dB values at each time bin.

    Parameters
    ----------
    time_bins : array of shape (n_windows,) — time bin centres (ms).
    freq_bins : array of shape (n_freq_bins,) — frequency bin centres (Hz).
    stft_db   : array of shape (n_windows, n_freq_bins) — magnitude in dB.
    path      : output CSV file path.
    """
    time_bins = np.asarray(time_bins)
    freq_bins = np.asarray(freq_bins)
    stft_db = np.asarray(stft_db)

    n_time = len(time_bins)
    n_freq = len(freq_bins)

    if stft_db.shape != (n_time, n_freq):
        raise ValueError(
            f"stft_db shape {stft_db.shape} does not match "
            f"(n_time={n_time}, n_freq={n_freq})."
        )

    header = "Frequency_Hz," + ",".join(f"{t:.4f}" for t in time_bins)

    rows = []
    for i, f in enumerate(freq_bins):
        row_values = stft_db[:, i]  # all time bins for this frequency
        row_str = ",".join(f"{v:.4f}" for v in row_values)
        rows.append(f"{f:.4f},{row_str}")

    csv_content = header + "\n" + "\n".join(rows)

    with open(path, "w", encoding="utf-8") as fh:
        fh.write(csv_content)
