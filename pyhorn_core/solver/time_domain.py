"""
Time-domain analysis and impulse-response tools.

Mirrors Hornresp time-domain outputs (pages 96–97): impulse response,
cumulative spectral decay (waterfall / CSD), and WAV export.

The core transform is an IFFT from the complex frequency-domain pressure
spectrum (``SPL + phase``) to the time domain.  The CSD is built by
progressively gating the impulse response and re-transforming — resonant
modes appear as ridges that persist across time slices.

Public API
----------
compute_impulse_response()   — IFFT of complex pressure → time series
compute_csd()                 — cumulative spectral decay waterfall
export_impulse_to_wav()      — write impulse response as 16-bit mono WAV
"""
import numpy as np
import wave
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TimeDomainResult:
    time_ms: np.ndarray       # time axis in milliseconds
    impulse: np.ndarray       # impulse response (Pa)
    step: np.ndarray          # step response (Pa·s)
    csd_times_ms: np.ndarray  # CSD time axis (ms)
    csd_freqs: np.ndarray     # CSD frequency axis (Hz)
    csd_db: np.ndarray        # CSD magnitude (dB), shape (n_times, n_freqs)


def compute_impulse_response(
    freqs: np.ndarray,
    pressure: np.ndarray,
    window_ms: float = 20.0,
) -> tuple:
    """
    Compute impulse response from complex pressure spectrum via IFFT.

    Args:
        freqs: frequency array (Hz), must be uniformly spaced
        pressure: complex pressure at each frequency
        window_ms: time window length in ms for the output

    Returns:
        (time_ms, impulse_response) arrays
    """
    df = freqs[1] - freqs[0]
    n_fft = int(round(1.0 / df / (1e-3)))  # samples for 1 ms resolution
    n_fft = max(n_fft, len(freqs) * 4)     # at least 4x zero-pad for interpolation
    # Make power of 2 for FFT efficiency
    n_fft = int(2 ** np.ceil(np.log2(n_fft)))

    # Build one-sided spectrum and IFFT
    # freqs covers f_min to f_max; we need 0 to f_max
    f_max = freqs[-1]
    full_spectrum = np.zeros(n_fft // 2 + 1, dtype=complex)

    # Map frequency bins
    freq_indices = np.round(freqs / df).astype(int)
    for i, fi in enumerate(freq_indices):
        if 0 <= fi < len(full_spectrum):
            full_spectrum[fi] = pressure[i]

    # IRFFT gives real-valued impulse response
    impulse = np.fft.irfft(full_spectrum, n=n_fft)

    dt = 1.0 / (n_fft * df)
    t = np.arange(n_fft) * dt * 1000.0  # ms

    # Trim to window
    n_window = int(window_ms / (dt * 1000.0))
    n_window = min(n_window, n_fft)

    return t[:n_window], impulse[:n_window]


def compute_csd(
    freqs: np.ndarray,
    pressure: np.ndarray,
    n_slices: int = 30,
    window_ms: float = 20.0,
) -> TimeDomainResult:
    """
    Compute Cumulative Spectral Decay (waterfall) from complex pressure spectrum.

    Uses progressively shorter time windows to show how energy at each frequency
    decays over time. A resonance that rings appears as a ridge persisting across
    multiple time slices.

    Args:
        freqs: uniformly spaced frequency array (Hz)
        pressure: complex pressure at each frequency
        n_slices: number of time slices in the waterfall
        window_ms: total time window (ms)

    Returns:
        TimeDomainResult with all time and frequency domain data
    """
    df = freqs[1] - freqs[0]

    # Compute full impulse response first
    t_full, impulse_full = compute_impulse_response(freqs, pressure, window_ms)
    dt = t_full[1] - t_full[0] if len(t_full) > 1 else 1.0
    n_samples = len(impulse_full)

    # Step response = cumulative integral of impulse
    # impulse is in Pa·s (from irfft of pressure spectrum)
    # dt is in ms; convert to seconds: dt_s = dt (ms) * 1e-3
    step = np.cumsum(impulse_full) * (dt * 1e-3)

    # CSD: apply progressively shorter windows
    slice_times = np.linspace(0, window_ms, n_slices)
    n_csd_fft = int(2 ** np.ceil(np.log2(n_samples)))
    csd_freqs = np.fft.rfftfreq(n_csd_fft, d=dt / 1000.0)

    # Trim CSD frequencies to the simulation range
    f_mask = (csd_freqs >= freqs[0]) & (csd_freqs <= freqs[-1])
    csd_freqs_trimmed = csd_freqs[f_mask]

    csd_db = np.zeros((n_slices, len(csd_freqs_trimmed)))

    for i, t_start in enumerate(slice_times):
        # Zero out samples before t_start (simulate gating)
        n_start = int(t_start / dt) if dt > 0 else 0
        n_start = min(n_start, n_samples)

        gated = np.zeros(n_samples)
        gated[n_start:] = impulse_full[n_start:]

        # Apply half-Hann fade-in at the gate edge to reduce spectral splatter
        fade_len = min(32, n_samples - n_start)
        if fade_len > 1:
            fade = 0.5 * (1.0 - np.cos(np.pi * np.arange(fade_len) / fade_len))
            gated[n_start:n_start + fade_len] *= fade

        spectrum = np.fft.rfft(gated, n=n_csd_fft)
        magnitude = np.abs(spectrum[f_mask])
        csd_db[i] = 20.0 * np.log10(magnitude + 1e-12)

    return TimeDomainResult(
        time_ms=t_full,
        impulse=impulse_full,
        step=step,
        csd_times_ms=slice_times,
        csd_freqs=csd_freqs_trimmed,
        csd_db=csd_db,
    )


def export_impulse_to_wav(
    freqs: np.ndarray,
    spl: np.ndarray,
    phase: np.ndarray,
    output_path: Path | str,
    fs: int = 44100,
    window_ms: float = 50.0,
) -> None:
    """
    Compute impulse response from SPL + phase data and export as 16-bit PCM WAV.

    Mirrors Hornresp WAV export (see Hornresp page 96 — impulse response output).

    Args:
        freqs: uniformly spaced frequency array (Hz)
        spl: SPL values in dB re 2e-5 Pa
        phase: unwrapped phase in radians
        output_path: path for the output .wav file
        fs: sample rate in Hz (default 44100)
        window_ms: time window for the impulse response in ms (default 50 ms)
    """
    # Reconstruct complex pressure spectrum from SPL (dB ref 2e-5 Pa) and phase
    p_ref = 2e-5
    pressure = (10 ** (spl / 20.0)) * p_ref * np.exp(1j * phase)

    # Compute impulse response
    t_ms, impulse = compute_impulse_response(freqs, pressure, window_ms=window_ms)

    # Convert to seconds and resample to target sample rate
    t_s = t_ms * 1e-3
    dt_s = 1.0 / fs

    # Build the continuous-time impulse response sampled at fs
    # Trim to window; pad to at least 1 sample
    t_end = t_s[-1] if len(t_s) > 0 else window_ms * 1e-3
    n_samples = max(int(t_end * fs) + 1, fs // 20)  # at least 50ms
    t_new = np.arange(n_samples) * dt_s

    # Interpolate impulse to target sample rate
    if len(t_s) >= 2:
        impulse_resampled = np.interp(t_new, t_s, impulse)
    else:
        impulse_resampled = np.zeros(n_samples)

    # Normalize to prevent 16-bit PCM clipping (scale so max amplitude = 0.99)
    peak = np.max(np.abs(impulse_resampled))
    if peak > 0:
        impulse_resampled = impulse_resampled * (32767 * 0.99 / peak)

    # Convert to 16-bit signed integers
    samples_i16 = np.clip(impulse_resampled, -32768, 32767).astype(np.int16)

    # Write WAV file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(fs)
        wf.writeframes(samples_i16.tobytes())