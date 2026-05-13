"""Shared acoustic scoring primitives used by optimization layers."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ResponseMetrics:
    """Acoustic summary metrics computed from an SPL response."""

    mean_spl: float
    flatness_db: float
    bass_mean_spl: float | None
    bass_deficit_db: float
    cutoff_frequency_hz: float


def spl_in_band(
    spl: np.ndarray,
    freq: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """Return SPL samples between two frequency bounds, inclusive."""
    return spl[(freq >= low_hz) & (freq <= high_hz)]


def mean_spl_in_band(
    spl: np.ndarray,
    freq: np.ndarray,
    low_hz: float,
    high_hz: float,
) -> float | None:
    """Return the mean SPL in a band, or None when no samples are present."""
    band_spl = spl_in_band(spl, freq, low_hz, high_hz)
    if len(band_spl) == 0:
        return None
    return float(np.mean(band_spl))


def estimate_cutoff_frequency(spl: np.ndarray, freq: np.ndarray, f_min: float) -> float:
    """Estimate the -3 dB cutoff frequency from the SPL curve."""
    band_spl = spl_in_band(spl, freq, f_min * 2.0, f_min * 5.0)
    if len(band_spl) == 0:
        band_spl = spl_in_band(spl, freq, f_min * 1.5, f_min * 4.0)
    if len(band_spl) == 0:
        band_spl = spl_in_band(spl, freq, f_min, float(freq[-1]))
    if len(band_spl) == 0:
        return f_min

    avg_passband = float(np.mean(band_spl))
    threshold = avg_passband - 3.0

    search_freq = freq[freq >= f_min]
    search_spl = spl[freq >= f_min]
    if len(search_freq) == 0:
        return f_min

    below = np.where(search_spl <= threshold)[0]
    if below.size == 0:
        return float(search_freq[-1])

    return float(search_freq[below[0]])


def cutoff_penalty(f_c: float, f_min: float) -> float:
    """Soft penalty for the cutoff constraint.

    Piecewise sigmoid-like decay:
    - f_c <= f_min          → 1.0 (full credit, no penalty)
    - f_c >= f_min × 2.5    → 0.0 (zero score — completely fails)
    - Smooth sigmoid falloff between those points

    The previous hard binary cutoff (1.0 vs 0.0 at f_min × 1.5) discarded
    potentially useful designs too aggressively. A soft penalty lets the
    Pareto front retain designs that are slightly over cutoff but
    otherwise well-shaped.
    """
    if f_c <= f_min:
        return 1.0
    if f_c >= f_min * 2.5:
        return 0.0
    # Smooth sigmoid-like decay (steepest near f_min × 1.5)
    t = (f_c - f_min) / (f_min * 1.5)  # 0→1 over the transition zone
    return max(0.0, 1.0 - _sigmoid(t, steepness=4.0))


def _sigmoid(t: float, steepness: float = 4.0) -> float:
    """Standard sigmoid centered at t=0.5, range [0, 1]."""
    # 1/(1 + exp(-steepness*(t-0.5)))
    return 1.0 / (1.0 + np.exp(-steepness * (t - 0.5)))


def compute_response_metrics(
    spl: np.ndarray,
    freq: np.ndarray,
    f_min: float,
    f_max: float,
) -> ResponseMetrics:
    """Compute shared acoustic summary metrics for a response curve."""
    band_spl = spl_in_band(spl, freq, f_min, f_max)
    if len(band_spl) == 0:
        mean_spl = 0.0
        flatness_db = 0.0
    else:
        mean_spl = float(np.mean(band_spl))
        flatness_db = float(np.std(band_spl))

    bass_spl = spl_in_band(spl, freq, f_min, f_min * 2.0)
    bass_mean_spl = float(np.mean(bass_spl)) if len(bass_spl) > 0 else None
    bass_deficit_db = (
        max(0.0, mean_spl - bass_mean_spl) if bass_mean_spl is not None else 0.0
    )

    return ResponseMetrics(
        mean_spl=mean_spl,
        flatness_db=flatness_db,
        bass_mean_spl=bass_mean_spl,
        bass_deficit_db=float(bass_deficit_db),
        cutoff_frequency_hz=estimate_cutoff_frequency(spl, freq, f_min),
    )
