"""
pyhorn_physics.radiation — piston radiation impedance and directivity.

Purpose
=======
This module provides the acoustic radiation boundary condition for horn and
enclosure simulations.  It computes the complex radiation impedance
(Z_rad = R_rad + j·X_rad) seen by a vibrating piston (driver cone or horn
mouth) radiating into a finite or infinite baffle, as well as the
Frequency-Dependent Directivity (FDD) model used for off-axis response
prediction.

In the BLH signal cascade the radiation impedance is applied **after** the
TMM throat-to-mouth cascade::

    Driver cone
       │
       ├── Z_front_load = Z_rad_front  (half-space radiation)
       │
       └── Z_rear_load = Z_throat + Z_ab  (horn path + rear chamber)
              │
              └── TMM cascade (throat → mouth)
                     │
                     └── Z_rad_mouth  (mouth radiation, this module)

Hornresp Pages Referenced
==========================
======= ===================================================================
Page    Topic
======= ===================================================================
  11    Radiation impedance — circular piston in infinite baffle
  12    Semi-inductance Le(f) model (used by infinite_baffle_response)
  13    Direct radiation and combined front/horn loading
  77    FDD directivity model; Lossy Le model
  92    FDD model equations
======= ===================================================================

Physical Constants
==================
============ ==============================================================
Constant    Value                         Description
============ ==============================================================
``RHO``    1.21 kg/m³                    Air density at ~20 °C
``C``      343.0 m/s                     Speed of sound in air
``Z0``     ≈ 415.0 Pa·s/m³ (= RHO·C)    Characteristic acoustic impedance
============ ==============================================================

Public API
==========
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``radiation_impedance(freq, mouth_area, ang[, mouth_w, mouth_h, _Zc, _a])``
     - Radiation impedance of a circular or rectangular piston in a baffle.
       Uses Levine/Inglis (full Bessel/Struve) for ka ≥ 0.1, Rayleigh
       approximation for ka < 0.1.  Rectangular piston uses the Morse & Ingard
       k²·S² low-ka formula.  ang = π (half-space) or 2π (full-space).
   * - ``infinite_baffle_response(freqs, driver)``
     - Natural SPL of a bare driver on an infinite baffle (both sides half-space).

Helper Functions (semi-public — not usually called directly)
=============================================================
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``_circular_piston_radiation_impedance(freq, mouth_area, ang, Zc, a)``
     - Levine/Inglis exact radiation impedance (full Bessel/Struve, all ka).
   * - ``_fdd_directivity_index(freqs, mouth_area[, f_c, D_max])``
     - FDD directivity index (dB).  Smooth omni→directional transition
       based on mouth ka.  Hornresp pages 77/92.
   * - ``_fdd_off_axis_spl(freqs, mouth_area, angles[, f_c, D_max])``
     - Off-axis SPL (dB rel. on-axis) using the FDD piston directivity factor.
   * - ``_fdd_radiation_angle(freqs, mouth_area, off_axis_spl, angles[, f_c])``
     - Mean -6 dB beamwidth half-angle from FDD off-axis data.
   * - ``_miki_factors(freq, sigma)``
     - Miki (1990) Zc and k correction factors for wall flow resistivity σ.

References
==========
- Levine & Schwinger (1950): exact circular piston radiation impedance.
- Morse & Ingard §9.2–9.3 (1968): rectangular and circular piston formulas.
- Beranek (1949): acoustic impedance relations.
- Miki, Y. (1990). *Journal of the Acoustical Society of Japan* 11(1), 19–24.
- Hornresp manual, pages 11–13, 77, 92.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple
import numpy as np
from scipy.special import jv, struve

if TYPE_CHECKING:
    from pyhorn_core.config.models import DriverSpecs

# ─── Physical constants ───────────────────────────────────────────────────────
RHO = 1.21  # Air density kg/m³
C = 343.0  # Speed of sound m/s
Z0 = RHO * C  # Characteristic impedance of air

# ─── Miki absorption factor clamping bounds ────────────────────────────────────
# The Miki (1990) model is only validated for 0.01 < f/sigma < 1.0.
# Outside this range the power-law formulas produce unphysically large values.
_MIKI_REAL_MIN = 0.7
_MIKI_REAL_MAX = 2.0
_MIKI_IMAG_MIN = -1.0
_MIKI_IMAG_MAX = 0.0


# ─── Radiation impedance ────────────────────────────────────────────────────────

def _miki_factors(freq: float, sigma: float) -> Tuple[complex, complex]:
    """
    Miki (1990) frequency-dependent absorption model.
    Returns (Zc_factor, k_factor) — complex multipliers for characteristic
    impedance and wavenumber respectively.

    sigma: flow resistivity in N·s/m⁴ (= Rayls/m = Pa·s/m²).
    Valid for 0.01 < f/sigma < 1.0.

    Ref: Miki, Y. (1990). JASJ 11(1), 19-24.
    """
    if sigma <= 0.0:
        # sigma=0 (perfectly hard wall) → no absorption modification
        return (1.0 + 0.0j, 1.0 + 0.0j)

    X = 1e3 * freq / sigma  # dimensionless frequency ratio
    Zc_factor = (1.0 + 5.50 * X**-0.632) - 1j * (8.43 * X**-0.632)
    k_factor = (1.0 + 7.81 * X**-0.618) - 1j * (11.41 * X**-0.618)

    Zc_factor = complex(
        np.clip(Zc_factor.real, _MIKI_REAL_MIN, _MIKI_REAL_MAX),
        np.clip(Zc_factor.imag, _MIKI_IMAG_MIN, _MIKI_IMAG_MAX),
    )
    k_factor = complex(
        np.clip(k_factor.real, _MIKI_REAL_MIN, _MIKI_REAL_MAX),
        np.clip(k_factor.imag, _MIKI_IMAG_MIN, _MIKI_IMAG_MAX),
    )
    return Zc_factor, k_factor


def _circular_piston_radiation_impedance(
    freq: float, mouth_area: float, ang: float, Zc: float, a: float
) -> complex:
    """
    Exact radiation impedance for a rigid circular piston in an infinite baffle.
    Uses the analytical Bessel and Struve function formulas, valid for all ka.

    Z_rad = R_rad + j·X_rad

    where:
        R_rad = Zc · [1 − 2·J₁(2ka) / (2·ka)]
        X_rad = Zc · H₁(2ka) / ka

    Small-ka limits:
        R_rad → Zc · (ka)²/2   (O(ka²))
        X_rad → Zc · 8ka/3π    (O(ka))

    Ref: Beranek (1954) Acoustics; Kinsler & Frey.
    """
    k = 2 * np.pi * freq / C
    ka = k * a

    if ka < 1e-6:
        return 0.0 + 0.0j

    if ka < 0.1:
        R_rad = Zc * (ka ** 2) / 2.0
        X_rad = Zc * 8.0 * ka / (3.0 * np.pi)
    else:
        # 2*J1(x)/x
        j1_term = 2.0 * jv(1, 2.0 * ka) / (2.0 * ka)
        R_rad = Zc * (1.0 - j1_term)
        # H1(x) / (x/2) = 2*H1(x)/x = H1(2ka)/ka
        X_rad = Zc * struve(1, 2.0 * ka) / ka

    R_rad *= (2.0 * np.pi) / ang
    X_rad *= (2.0 * np.pi) / ang

    return R_rad + 1j * X_rad


def radiation_impedance(
    freq: float,
    mouth_area: float,
    ang: float,
    _Zc: float | None = None,
    _a: float | None = None,
    mouth_width: float | None = None,
    mouth_height: float | None = None,
) -> complex:
    """
    Radiation impedance of a circular or rectangular piston in a baffle.

    Uses Levine/Inglis (full Bessel/Struve) for ka ≥ 0.1, Rayleigh
    approximation for ka < 0.1.  Rectangular piston uses the Morse & Ingard
    k²·S² low-ka formula.

    ang = π (half-space, e.g. front baffle) or 2π (full-space).
    """
    k = 2 * np.pi * freq / C
    if mouth_area <= 0.0:
        return 0.0j
    Zc = _Zc if _Zc is not None else Z0 / mouth_area

    if (
        mouth_width is not None
        and mouth_height is not None
        and mouth_width > 0
        and mouth_height > 0
    ):
        R_rad = Zc * k**2 * mouth_area**3 / (2.0 * np.pi) * (2.0 * np.pi / ang)
        X_rad = Zc * k * (mouth_width + mouth_height) * mouth_area / (3.0 * np.pi)
        return R_rad + 1j * X_rad
    a = _a if _a is not None else np.sqrt(mouth_area / np.pi)
    return _circular_piston_radiation_impedance(freq, mouth_area, ang, Zc, a)


# ─── Frequency-Dependent Directivity (FDD) ────────────────────────────────────

def _fdd_directivity_index(
    freqs: np.ndarray,
    mouth_area: float,
    f_c: float = 300.0,
    D_max: float = 5.0,
) -> np.ndarray:
    """
    Frequency Dependent Directivity (FDD) model.

    Provides a smooth transition from omnidirectional radiation at low frequencies
    to increasingly directional radiation at high frequencies, based on the horn
    mouth dimensions.

    The directivity index (DI) in dB is::

        DI(f) = D_max × [1 − exp(−(f / f_c)²)]

    Parameters
    ----------
    freqs      : np.ndarray — frequency points (Hz)
    mouth_area : float — effective radiating area of the horn mouth (m²)
    f_c        : float — characteristic transition frequency in Hz (default 300 Hz)
    D_max      : float — maximum directivity index in dB (default 5.0 dB)

    Returns
    -------
    np.ndarray — directivity index in dB at each frequency.

    Ref: Hornresp pages 77, 92.
    """
    a = np.sqrt(mouth_area / np.pi)
    ka = 2.0 * np.pi * a * freqs / C

    transition = 1.0 - np.exp(-(freqs / f_c) ** 2)
    di_linear = D_max * transition
    return np.clip(di_linear, 0.0, D_max)


def _piston_directivity_factor(ka: float, sin_theta: float) -> float:
    """
    Levine/Inglis piston directivity factor for a circular piston in an infinite baffle.

        D(θ) = [2·J₁(ka·sin(θ)) / (ka·sin(θ))]²

    At low ka (ka → 0), D(θ) → 1 (omnidirectional).
    At high ka, D(θ) narrows — piston becomes increasingly directional.

    Parameters
    ----------
    ka       : float — wave number × piston radius (ka = 2π·a·f/C)
    sin_theta: float — sin of the off-axis angle

    Returns
    -------
    float — directivity factor in [0, 1]
    """
    x = ka * sin_theta
    if abs(x) < 1e-6:
        return 1.0  # limiting case: on-axis or ka→0
    return (2.0 * jv(1, x) / x) ** 2


def _fdd_off_axis_spl(
    freqs: np.ndarray,
    mouth_area: float,
    angles: np.ndarray,
    f_c: float = 300.0,
    D_max: float = 5.0,
) -> np.ndarray:
    """
    Off-axis SPL (dB relative to on-axis) using the FDD piston directivity model.

    Uses the Levine/Inglis piston directivity factor which depends on ka
    (and therefore on mouth_area), giving physically correct directivity:
    larger mouths → narrower beamwidth at the same frequency.

    Parameters
    ----------
    freqs      : np.ndarray — frequency points (Hz)
    mouth_area : float — effective mouth area (m²)
    angles     : np.ndarray — off-axis angles in degrees
    f_c        : float — FDD characteristic transition frequency (Hz)
    D_max      : float — maximum directivity index (dB) [used for HF transition]

    Returns
    -------
    np.ndarray — off-axis SPL in dB relative to on-axis,
                 shape (len(freqs), len(angles)).
    """
    a = np.sqrt(mouth_area / np.pi)
    ka_vals = 2.0 * np.pi * a * freqs / C  # shape (n_freq,)

    # Transition factor: 0 (omni) at low f → 1 (full piston directivity) at HF
    # Uses Hornresp FDD f_c as the characteristic frequency.
    freqs_arr = np.asarray(freqs)
    transition = 1.0 - np.exp(-(freqs_arr / f_c) ** 2)

    n_freq = len(freqs)
    n_angles = len(angles)
    result = np.zeros((n_freq, n_angles), dtype=float)

    for j, ang_deg in enumerate(angles):
        ang_rad = np.radians(ang_deg)
        sin_theta = np.sin(ang_rad)

        for i in range(n_freq):
            ka = float(ka_vals[i])
            # Levine/Inglis piston directivity factor (depends on ka and thus mouth_area)
            D_piston = _piston_directivity_factor(ka, sin_theta)
            # Transition from omni (D=1) at LF to piston directivity at HF
            D_fdd = 1.0 - transition[i] * (1.0 - D_piston)
            # Clip and convert to dB relative to on-axis (D_fdd(on-axis) = 1 → 0 dB)
            rel_spl = 10.0 * np.log10(max(D_fdd, 1e-12))
            result[i, j] = rel_spl

    return result


def _fdd_radiation_angle(
    freqs: np.ndarray,
    mouth_area: float,
    off_axis_spl: np.ndarray,
    angles: np.ndarray,
    f_c: float = 300.0,
) -> Optional[float]:
    """
    Compute mean -6 dB beamwidth half-angle using the FDD model.

    For each frequency, finds the angle where directivity drops to -6 dB
    relative to on-axis. Returns the mean of those angles across the band
    (excluding near-omnidirectional frequencies where beamwidth > 88°).

    Parameters
    ----------
    freqs       : np.ndarray — frequency points (Hz)
    mouth_area  : float — mouth area (m²)
    off_axis_spl: np.ndarray — FDD off-axis SPL (dB rel, shape (n_freq, n_angles))
    angles      : np.ndarray — corresponding angles in degrees
    f_c         : float — FDD characteristic frequency

    Returns
    -------
    Mean -6 dB beamwidth half-angle in degrees, or None if insufficient data.
    """
    if off_axis_spl.shape[1] < 2:
        return None

    angles_6db = []
    for i in range(len(freqs)):
        rel_spl_at_angle = off_axis_spl[i, :]
        below_6db = np.where(rel_spl_at_angle <= -6.0)[0]
        if len(below_6db) > 0:
            idx_below = below_6db[0]
            if idx_below == 0:
                ang_6db = float(angles[0])
            else:
                idx_above = idx_below - 1
                s1 = float(angles[idx_above])
                s2 = float(angles[idx_below])
                v1 = rel_spl_at_angle[idx_above]
                v2 = rel_spl_at_angle[idx_below]
                if abs(v2 - v1) > 1e-9:
                    ang_6db = s1 + (s2 - s1) * (-6.0 - v1) / (v2 - v1)
                else:
                    ang_6db = s1
            angles_6db.append(ang_6db)

    if not angles_6db:
        return None

    valid = [a for a in angles_6db if a < 88.0]
    if not valid:
        return None
    return float(np.mean(valid))
