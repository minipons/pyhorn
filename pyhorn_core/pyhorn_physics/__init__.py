"""
pyhorn_physics — acoustic physics primitives for pyhorn.

This package provides the low-level acoustic physics layer used by the
pyhorn solver.  It is intentionally decoupled from config and I/O so that
the same primitives can be used by the CLI, the API server, and the desktop
UI without pulling in heavy dependencies.

Architecture
============
The package is divided into three logical "sub-modules" that live in this
single ``__init__.py`` for historical reasons (they were extracted from
``solver/models.py`` on 2026-05-02 before being split):

  radiation  — piston radiation, directivity, and radiation impedance
  tmm        — Transfer-Matrix Method primitives (tube, step, bend, compliance)
  driver     — driver motor physics (Bl, Le, impedance, excursion)

All physics primitives are stateless functions; no config objects are
instantiated here.  Type hints use ``TYPE_CHECKING`` imports to avoid
circular dependencies.

Physical Constants
==================
============ ==============================================================
Constant    Value                         Description
============ ==============================================================
``RHO``    1.21 kg/m³                    Air density at ~20 °C
``C``      343.0 m/s                     Speed of sound in air
``Z0``     ≈ 415.0 Pa·s/m³ (= RHO·C)    Characteristic acoustic impedance
============ ==============================================================

Miki (1990) Wall-Absorption Model
==================================
The Miki model computes frequency-dependent correction factors for
characteristic impedance (Zc_factor) and wavenumber (k_factor) inside a
porous horn wall (flow resistivity σ in Rayls/m).  Both factors are clamped
to physically reasonable envelopes outside the validated range
(0.01 < f/σ < 1.0) to prevent the power-law formulas from producing
unphysically large values at bass frequencies.

Reference: Miki, Y. (1990). *Journal of the Acoustical Society of Japan*
11(1), 19–24.

Public API — Radiation
=======================
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``radiation_impedance(freq, mouth_area, ang[, mouth_w, mouth_h])``
     - Radiation impedance of a circular or rectangular piston in a baffle.
       Uses Levine/Inglis (full Bessel/Struve) for ka ≥ 0.1, Rayleigh
       approximation for ka < 0.1.  Rectangular piston uses the Morse & Ingard
       k²·S² low-ka formula.  ang = π (half-space) or 2π (full-space).
   * - ``infinite_baffle_response(freqs, driver)``
     - Natural SPL of a bare driver on an infinite baffle (both sides half-space).

Public API — TMM Transfer-Matrix Primitives
============================================
All matrices are returned as 2×2 complex numpy arrays.  Cascading multiplies
the leftmost matrix against the next-to-the-right (``result @ m``), matching
the acoustic wave-travel direction.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``tube_segment_tmatrix(freq, length, area[, fr])``
     - Lossless or lossy (Miki) uniform tube section.  The characteristic
       impedance is Zc = Z0/A and the propagation constant is k = ω/C (optionally
       modified by the Miki factors when fr > 0).
   * - ``area_step_tmatrix(freq, area1, area2[, lem_model])``
     - Area discontinuity (expansion or contraction).  ``lem_model="basic"``
       adds a series inertance and shunt compliance to approximate corner-pocket
       losses in stair-step horns.  Default is ideal (no loss).
   * - ``bend_tmatrix(freq, area, angle_rad)``
     - Duct bend modelled as a series length-correction impedance and a shunt
       corner-dead-volume compliance.  angle_rad = 0 → identity; π → 180° U-turn.
   * - ``compliance_tmatrix(freq, volume[, fr, area])``
     - Acoustic compliance of a sealed chamber (closed-box stiffness).  When
       fr > 0 the chamber is modelled as a short lossy tube using the Miki
       absorption model.
   * - ``throat_adapter_tmatrix(horn, freq[, fr])``
     - T-matrix for the throat adapter (profiled transition duct between the
       throat chamber and horn proper).  Supports cylindrical, conical,
       exponential, and parabolic profiles via N = 8 sub-segments.
   * - ``cascade(matrices)``
     - Fold a list of 2×2 T-matrices into a single equivalent matrix.

Public API — Acoustic Impedance
===============================
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``rear_chamber_impedance(freq, volume, length[, fr, chamber_type, …])``
     - Acoustic impedance of the rear loading chamber.  Supports three types:
       ``"sealed"`` (closed tube, default), ``"vented"`` (Helmholtz resonator),
       ``"coupling"`` (pure stiffness + annular throat radiation, BLH coupling
       chamber model).  Hornresp pages 62–65.
   * - ``slavbas_impedance(freq, vrc, rleak)``
     - Slavic "slave bass" rear-chamber: sealed box with a resistive leak vent
       in parallel with the compliance.  Smooth rolloff without a vented-box
       resonance peak.  Hornresp page 65.
   * - ``vented_box_impedance(freq, vrc, lrc, fr_tuning[, ql])``
     - Bass-reflex / Helmholtz resonator model.  Derives port area from the
       tuning frequency using iterated end-correction.
   * - ``passive_radiator_impedance(freq, vrc, mma, sp[, ql_pr])``
     - Passive radiator system impedance.

Public API — Driver Physics
===========================
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``_le_freq_dependent(le_const, f, f_ref)``
     - Frequency-dependent voice-coil inductance Le(f) = Le₀·√(1 + (f/f_ref)²).
       Hornresp page 12 (semi-inductance model).
   * - ``_lossy_le_impedance(f, R_e_eddy, f_ref)``
     - Eddy-current loss resistance R_lossy(f) = R_e_eddy·(f/f_ref)².
       Hornresp page 77 (Lossy Le model).
   * - ``_driver_impedance(bl, re, le, mms, cms, rms, sd, f[, …])``
     - Returns (Z_e, Z_ms, Z_me, Z_mt): electrical, suspension, motor-reflected,
       and total mechanical impedances at frequency f.
   * - ``_velocity(P_source, Z_tot)``
     - Diaphragm volume velocity U = P_source / Z_tot.
   * - ``_displacement(v_driver, f)``
     - Complex displacement x = v / (jω).
   * - ``_excursion(x_driver)``
     - Peak-to-peak excursion in mm from complex displacement (×2√2 for RMS→pp).

Helper Functions (not usually called directly)
===============================================
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function
     - Description
   * - ``_miki_factors(freq, sigma)``
     - Miki (1990) Zc and k correction factors for flow resistivity σ.
   * - ``_circular_piston_radiation_impedance(freq, mouth_area, ang, Zc, a)``
     - Levine/Inglis exact radiation impedance (Bessel/Struve).
   * - ``_merge_small_throat_segments(segments, …)``
     - Merges very short throat segments (len < 2 mm) to suppress TMM
       numerical artifacts (e.g. the ~1847 Hz Hiro resonance).
   * - ``_is_single_segment_horn(horn)``
     - Returns True when the horn has exactly one radiating element
       (needed for second-tone distortion calculation).
   * - ``_compute_second_tone_distortion(freqs, driver_bl, …)``
     - Analytical 2nd-harmonic distortion based on compliance non-linearity
       (α ≈ 0.3 m⁻¹).  Valid for mass-dominated single-segment horns above fs.
   * - ``_fdd_directivity_index(freqs, mouth_area[, f_c, D_max])``
     - Frequency-Dependent Directivity model.  Smooth transition from omni
       to directional based on mouth ka.  Hornresp pages 77/92.
   * - ``_fdd_off_axis_spl(freqs, mouth_area, angles[, f_c, D_max])``
     - Off-axis SPL using the FDD piston directivity factor.
   * - ``_fdd_radiation_angle(freqs, mouth_area, off_axis_spl, angles)``
     - Mean -6 dB beamwidth half-angle from FDD off-axis data.
   * - ``_apply_notch_filter(freqs, spl, notch_frequencies[, notch_q])``
     - Gaussian notch filters at artifact frequencies (TMM numerical spikes).
   * - ``_smooth_spl_near_artifacts(freqs, spl, artifacts)``
     - Narrow symmetric moving average around flagged artifact bins.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np

# ─── Radiation impedance (extracted to pyhorn_physics.radiation) ───────────────
from pyhorn_core.pyhorn_physics.radiation import (
    RHO,
    C,
    Z0,
    _miki_factors,
    _circular_piston_radiation_impedance,
    radiation_impedance,
    _fdd_directivity_index,
    _fdd_off_axis_spl,
    _fdd_radiation_angle,
)  # noqa: E402

if TYPE_CHECKING:
    from pyhorn_core.config.models import DriverSpecs, HornGeometry

def tube_segment_tmatrix(
    freq: float, length: float, area: float, fr: float = 0.0
) -> np.ndarray:
    w = 2 * np.pi * freq
    k = w / C  # wavenumber
    Zc = Z0 / area  # characteristic acoustic impedance

    if fr > 0:
        Zc_f, k_f = _miki_factors(freq, fr)
        k = k * k_f
        Zc = Zc * Zc_f

    kL = k * length

    A = np.cos(kL)
    B = 1j * Zc * np.sin(kL)
    Cv = 1j * np.sin(kL) / Zc
    D = np.cos(kL)

    return np.array([[A, B], [Cv, D]])


def area_step_tmatrix(
    freq: float,
    area1: float,
    area2: float,
    lem_model: Optional[str] = None,
    lem_strength: float = 1.0,
    lem_resistance: float = 0.0,
) -> np.ndarray:
    ratio = area1 / area2

    if lem_model is None or lem_model.lower() in {"", "ideal", "none"}:
        return np.array([[1.0, 0.0], [0.0, ratio]], dtype=complex)

    model = lem_model.lower()
    if model != "basic":
        raise ValueError(f"Unknown LEM step model: {lem_model}")

    # Basic stair-step LEM: a local series inertance at all discontinuities and
    # a shunt compliance for expansions to approximate the corner pocket.
    w = 2 * np.pi * freq
    min_area = max(min(area1, area2), 1e-9)
    a1 = np.sqrt(area1 / np.pi)
    a2 = np.sqrt(area2 / np.pi)
    a_eff = 0.5 * (a1 + a2)
    delta = abs(area2 - area1) / max(area1, area2, 1e-9)

    delta_L = lem_strength * 0.35 * a_eff * delta
    R_series = lem_resistance * (Z0 / min_area) * delta**2
    Z_series = R_series + 1j * w * RHO * delta_L / min_area

    if area2 > area1:
        pocket_volume = lem_strength * 0.25 * (area2 - area1) * a_eff
        Y_shunt = 1j * w * pocket_volume / (RHO * C**2)
    else:
        Y_shunt = 0.0j

    T_step = np.array([[1.0, 0.0], [0.0, ratio]], dtype=complex)
    T_series = np.array([[1.0, Z_series], [0.0, 1.0]], dtype=complex)
    T_shunt = np.array([[1.0, 0.0], [Y_shunt, 1.0]], dtype=complex)
    return T_series @ T_shunt @ T_step


def bend_tmatrix(freq: float, area: float, angle_rad: float) -> np.ndarray:
    """
    Models a duct bend as a series impedance (effective length correction)
    plus a shunt compliance (corner dead-volume).
    angle_rad = 0 → straight (identity), angle_rad = pi → 180° U-turn (max effect).
    """
    if angle_rad < 0.01:
        return np.eye(2, dtype=complex)

    w = 2 * np.pi * freq
    k = w / C
    a_eff = np.sqrt(area / np.pi)

    # Series impedance: effective length correction at bend
    delta_L = 0.6 * a_eff * (angle_rad / np.pi)
    Z_series = 1j * k * (RHO * C / area) * delta_L

    # Shunt compliance: corner dead-volume
    V_corner = area * a_eff * (1.0 - np.cos(angle_rad / 2.0))
    Y_shunt = 1j * w * V_corner / (RHO * C**2)

    T_s = np.array([[1.0, Z_series], [0.0, 1.0]], dtype=complex)
    T_c = np.array([[1.0, 0.0], [Y_shunt, 1.0]], dtype=complex)
    return T_s @ T_c


def compliance_tmatrix(
    freq: float, volume: float, fr: float = 0.0, area: float = 0.0
) -> np.ndarray:
    w = 2 * np.pi * freq
    Ca = volume / (RHO * C**2)

    if fr == 0.0:
        return np.array([[1.0, 0.0], [1j * w * Ca, 1.0]], dtype=complex)
    else:
        if area > 0:
            A = area
            L = volume / area
        else:
            # Fallback: approximate the chamber as a cube
            L = volume ** (1.0 / 3.0)
            A = volume ** (2.0 / 3.0)

        # Model as a short lossy closed tube using Miki absorption
        Zc_f, k_f = _miki_factors(freq, fr)
        k = (w / C) * k_f
        Zc = (Z0 / A) * Zc_f
        kL = k * L
        if np.abs(kL) < 1e-4:
            # Low-frequency limit: pure compliance
            return np.array([[1.0, 0.0], [1j * w * Ca, 1.0]], dtype=complex)
        Z_shunt = -1j * Zc / np.tanh(kL)
        Y_shunt = 1.0 / Z_shunt

        return np.array([[1.0, 0.0], [Y_shunt, 1.0]], dtype=complex)


def cascade(matrices: List[np.ndarray]) -> np.ndarray:
    result = np.eye(2, dtype=complex)
    for m in matrices:
        result = result @ m
    return result


def _merge_small_throat_segments(
    segments: List[Tuple[float, ...]],
    bends: Optional[List[Tuple[float, float]]],
    bends_ext: Optional[List[Tuple[float, float, float]]],
    bend_positions: Optional[List[int]],
    n_throat: int = 4,
    min_len_merge: float = 0.005,
) -> List[Tuple[float, ...]]:
    """Merge very short segments near the throat into coarser ones.

    In area-tapered horn sections, the first few segments (closest to the
    throat) have the steepest area-ratio gradient.  Discretising them into many
    short sub-segments causes TMM numerical artifacts (e.g. the ~1847 Hz Hiro
    resonance).  This function merges the n_throat smallest segments provided
    their length is below min_len_merge, keeping total acoustic path length
    unchanged.
    """
    if not segments or len(segments) < n_throat + 1:
        return segments

    lengths = np.array([seg[0] for seg in segments])
    candidates = [
        i for i in range(min(n_throat, len(segments)))
        if lengths[i] < min_len_merge
    ]

    if not candidates:
        return segments

    first = candidates[0]
    last = candidates[-1]
    merged_length = float(np.sum(lengths[first : last + 1]))
    merged_area = float(np.mean([s[1] for s in segments[first : last + 1]]))
    merged_fr = float(
        np.mean([s[2] if len(s) >= 3 else 0.0 for s in segments[first : last + 1]])
    )

    merged_seg: Tuple[float, ...] = (merged_length, merged_area, merged_fr)
    new_segments = list(segments[:first]) + [merged_seg] + list(segments[last + 1 :])
    return new_segments


# ─── Acoustic Impedance functions ──────────────────────────────────────────────

def rear_chamber_impedance(
    freq: float,
    volume: float,
    length: float,
    fr: float = 0.0,
    chamber_type: str = "sealed",
    fr_tuning: float = 0.0,
    throat_area: float = 0.0,
) -> complex:
    """Acoustic impedance of the rear chamber.

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    volume : float
        Rear chamber volume in m³.
    length : float
        Rear chamber effective length in m.
        For ``chamber_type="sealed"``: used as the closed-tube length in the
        lossy compliance model.
        For ``chamber_type="vented"``: used as the port length (l_port) in the
        Helmholtz resonator model.
        For ``chamber_type="coupling"``: not used (pure stiffness model).
    fr : float
        Flow resistivity of damping material in Rayls/m (default 0 = lossless).
    chamber_type : str
        ``"sealed"`` (default): pure acoustic compliance / closed tube.
        ``"vented"``: vented-box / Helmholtz resonator model.  Produces an
        impedance peak at the rear-chamber resonance frequency.
        ``"coupling"``: pure acoustic compliance with radiation from the annular
        throat aperture — correct model for a BLH coupling chamber (no mass-
        controlled resonance peak).
    fr_tuning : float
        Helmholtz tuning frequency f_b (Hz) for ``chamber_type="vented"``.
        When > 0 the port area is derived to match this tuning.
        When 0, falls back to 55 Hz (legacy placeholder — set explicitly
        for geometry-specific calibration).
    throat_area : float
        Effective radiation area of the annular throat aperture (m²).  Used
        only for ``chamber_type="coupling"`` to compute the radiation impedance
        term.  Defaults to 0 (radiation term omitted) for backward compat.

    Returns
    -------
    complex
        Acoustic impedance Z_ab in Pa·s/m³ (acoustic ohms).
    """
    if volume <= 0:
        return 0.0j
    w = 2 * np.pi * freq

    # ── Coupling chamber (BLH pure-stiffness model) ─────────────────────────
    if chamber_type == "coupling" and volume > 0:
        # Pure stiffness: Z = 1/(jω·C_ab) + Z_rad_annulus
        # No mass term, no Helmholtz resonance peak.
        # C_ab = V_rc / (ρ·c²)
        C_ab = volume / (RHO * C**2)
        Z_coupling = -1j / (w * C_ab) if w > 0 else 0.0j

        # Radiation from the annular throat aperture
        if throat_area > 0:
            a_equiv = np.sqrt(throat_area / np.pi)
            Z_rad = radiation_impedance(freq, throat_area, 2.0 * np.pi, _a=a_equiv)
        else:
            Z_rad = 0.0j

        Z_ab = Z_coupling + Z_rad
        return Z_ab

    # ── Vented-box / Helmholtz resonator model ──────────────────────────────
    if chamber_type == "vented" and length > 0 and volume > 0:
        # Port area: derive from Helmholtz resonance equation assuming a
        # circular port.  We iterate to converge the end-correction:
        #   f_b = (c/2π) · √(A_port / (V_rc · L_eff))
        # where L_eff = l_port + 0.6·a_pipe (end-correction factor).
        #
        # Use explicit fr_tuning when provided; 55 Hz is a legacy placeholder.
        fb = fr_tuning if fr_tuning > 0 else 55.0

        vrc_m3 = volume
        lrc_m = length
        # Initial guess for port area (assuming L_eff ≈ 1.5·l_port):
        A_port = (2.0 * np.pi * fb / C) ** 2 * vrc_m3 * 1.5 * lrc_m
        A_port = max(A_port, 1e-6)

        # Iterate end correction (5 iterations is sufficient for convergence):
        for _ in range(5):
            a_pipe = np.sqrt(A_port / np.pi)
            L_eff = lrc_m + 0.6 * a_pipe
            A_port = (2.0 * np.pi * fb / C) ** 2 * vrc_m3 * L_eff
            A_port = max(A_port, 1e-6)

        # Port mass (acoustic mass of air in the port tube):
        #   M_v = ρ · L_eff / A_port
        a_pipe_final = np.sqrt(A_port / np.pi)
        L_eff_final = lrc_m + 0.6 * a_pipe_final
        M_v = RHO * L_eff_final / A_port

        # Box compliance:
        #   C_vb = V_rc / (ρ · c²)
        C_vb = vrc_m3 / (RHO * C**2)

        # Port radiation impedance:
        Z_rad_port = radiation_impedance(freq, A_port, 2.0 * np.pi)

        # Series impedance: mass + compliance + radiation
        #   Z_series = j·ω·M_v  +  1/(j·ω·C_vb)  +  Z_rad_port
        Z_mass = 1j * w * M_v
        Z_compliance = 1.0 / (1j * w * C_vb)

        Z_ab = Z_mass + Z_compliance + Z_rad_port
        return Z_ab

    # ── Sealed / closed-tube model (default) ───────────────────────────────
    # Pure acoustic compliance: Z_rc = 1 / (j·ω·C_rc)
    # C_rc = V_rc / (ρ·c²)
    Ca = volume / (RHO * C**2)
    return 1.0 / (1j * w * Ca) if w > 0 else 0.0j


def slavbas_impedance(
    freq: float,
    vrc: float,
    rleak: float,
) -> complex:
    """Acoustic impedance of a Slavic rear chamber (aperiodic / slave bass) box.

    The Slavic rear chamber is a sealed box with a resistive leak vent — an
    aperiodic box variant. The compliance (sealed volume) and leak resistance
    are in *parallel*:

        Z_sl = (1/(jωC_a)  ||  R_leak)  =  R_leak / (1 + jω·R_leak·C_a)

    where C_a = Vrc / (ρ·c²) is the acoustic compliance of the sealed volume.

    Key behaviours:
      - f → 0:   Z → 0  (the leak shorts DC pressure)
      - f → ∞:   Z → 1/(jωC_a)  (standard sealed-box rolloff)
      - Corner:  f_c = 1 / (2π·R_leak·C_a)  — overdamped, no resonance peak

    This gives a smooth rolloff without the boomy peak of a vented box, while
    extending low-frequency response compared to a tight sealed box.

    Reference: Hornresp manual page 65 — "Slavbas" (slave bass) rear chamber type.

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    vrc : float
        Sealed rear chamber volume in m³.
    rleak : float
        Acoustic leak resistance in N·s/m⁵ (Pa·s/m³).
        Can be derived from hole area and effective length:
            rleak = ρ·c·lrc / aleak²
        where ρ = 1.21 kg/m³, c = 343 m/s.

    Returns
    -------
    complex
        Acoustic impedance Z_sl in Pa·s/m³ (acoustic ohms).
    """
    if vrc <= 0:
        return 0.0j
    if rleak <= 0:
        # No leak → behaves as a standard sealed box (pure compliance)
        Ca = vrc / (RHO * C**2)
        w = 2.0 * np.pi * freq
        return 1.0 / (1j * w * Ca)

    Ca = vrc / (RHO * C**2)
    w = 2.0 * np.pi * freq

    # Parallel: Z = (Z_c · R) / (Z_c + R)  where Z_c = 1/(jωC_a)
    Zc = 1.0 / (1j * w * Ca)
    # Z_sl = (Zc || R) = (Zc · R) / (Zc + R)
    Z_sl = (Zc * rleak) / (Zc + rleak)
    return Z_sl


def vented_box_impedance(
    freq: float,
    vrc: float,
    lrc: float,
    fr_tuning: float,
    ql: float = 5.0,
) -> complex:
    """Acoustic impedance of a vented (bass-reflex) box.

    Models the vented box as a Helmholtz resonator: the port tube acts as an
    acoustic mass M_v in series with a compliance C_vb (the box volume), plus
    radiation impedance at the open port end.
    """
    if vrc <= 0 or lrc <= 0 or fr_tuning <= 0:
        return 0.0j

    w = 2 * np.pi * freq

    # Derive port area from tuning frequency (iterate for end correction)
    A_pipe = (2 * np.pi * fr_tuning / C) ** 2 * vrc * lrc
    A_pipe = max(A_pipe, 1e-6)

    for _ in range(5):
        a_pipe = np.sqrt(A_pipe / np.pi)
        L_eff = lrc + 0.6 * a_pipe
        A_pipe = (2 * np.pi * fr_tuning / C) ** 2 * vrc * L_eff
        A_pipe = max(A_pipe, 1e-6)

    M_v = RHO * L_eff / A_pipe
    C_vb = vrc / (RHO * C**2)

    R_leak = 0.0
    if ql > 0:
        R_leak = 1.0 / (2 * np.pi * fr_tuning * C_vb * ql)

    Z_rad_port = radiation_impedance(freq, A_pipe, 2.0 * np.pi)

    Z_mass = 1j * w * M_v
    Z_compliance = 1.0 / (1j * w * C_vb)

    Y_leak = 1.0 / (R_leak + 1e-12) if R_leak > 0 else 0.0j

    Z_series = Z_mass + Z_compliance + Z_rad_port
    Y_vb = Y_leak + 1.0 / (Z_series + 1e-12)
    return 1.0 / Y_vb


def passive_radiator_impedance(
    freq: float,
    vrc: float,
    mma: float,
    sp: float,
    ql_pr: float = 5.0,
) -> complex:
    """Acoustic impedance of a passive radiator (PR) system."""
    if vrc <= 0 or mma <= 0 or sp <= 0:
        return 0.0j

    w = 2 * np.pi * freq

    C_pr = vrc / (RHO * C**2)
    Z_pr = 1j * w * mma + 1.0 / (1j * w * C_pr)
    Y_box = 1j * w * C_pr

    R_leak = 0.0
    if ql_pr > 0:
        f_pr_sq = 1.0 / (mma * C_pr)
        f_pr_sq = max(f_pr_sq, 1e-12)
        f_pr = np.sqrt(f_pr_sq) / (2.0 * np.pi)
        R_leak = 1.0 / (2.0 * np.pi * f_pr * C_pr * ql_pr)

    Y_leak = 1.0 / (R_leak + 1e-12) if R_leak > 0 else 0.0j

    Y_total = Y_leak + 1.0 / (Z_pr + 1e-12) + Y_box
    return 1.0 / Y_total if abs(Y_total) > 1e-12 else 0.0j


def transmission_line_impedance(
    freq: float,
    ltl: float,
    area: float,
) -> complex:
    """Acoustic input impedance of a finite transmission line (closed at far end).

    Models a uniform pipe of length ``ltl`` with a **closed far end** (rigid
    termination).  The acoustic output is the radiation from the mouth (open end).

    The input impedance of a closed pipe (Hornresp page 091) is:

        Z_tl = j · Z0 · tan(k · ltl)

    where Z0 = ρ·c / A is the characteristic impedance and k = ω / c is the
    wavenumber.  At resonance (k·l = π/2, 3π/2, …) the pipe is in series
    resonance and Z_tl → 0; at anti-resonance (k·l = π, 2π, …) the pipe is
    in parallel resonance and Z_tl → ∞.

    Parameters
    ----------
    freq : float
        Frequency in Hz.
    ltl : float
        Acoustic length of the transmission line in metres (closed at far end).
    area : float
        Cross-sectional area of the transmission line in m².
        Used to compute the characteristic impedance Z0 = ρ·c / area.
        If <= 0, returns 0 (TL is disabled).

    Returns
    -------
    complex
        Acoustic input impedance Z_tl in Pa·s/m³ (acoustic ohms).
    """
    if ltl <= 0.0 or area <= 0.0:
        return 0.0j

    w = 2.0 * np.pi * freq
    if w <= 0.0:
        return 0.0j

    k = w / C
    Z0_tl = Z0 / area  # characteristic acoustic impedance of the line
    kL = k * ltl

    # Z_tl = j · Z0 · tan(k·l)
    # Use cot for the closed-pipe formula: Z_in = -j·Z0·cot(kL)
    # (Both are equivalent; cot = 1/tan.)
    # Guard against cot(kL) → ∞ at kL = nπ.
    tan_kL = np.tan(kL)
    if abs(tan_kL) < 1e-12:
        # kL ≈ nπ: cot → ∞ → Z_tl → 0 (series resonance, maximum velocity)
        return 0.0j

    Z_tl = 1j * Z0_tl * tan_kL
    return Z_tl


def throat_adapter_tmatrix(
    horn: "HornGeometry",
    freq: float,
    fr: float = 0.0,
) -> np.ndarray:
    """T-matrix for a throat adapter (transition duct between throat chamber and horn throat).

    Models the adapter as a short profiled section connecting the throat chamber
    (input end, area A0) to the horn throat (output end, area ap1) over axial length lpt.

    The profile type is read from ``horn.throat_adapter_type``:
      - ``cylindrical``  — constant area; standard lossless tube matrix
      - ``conical``       — linear area taper
      - ``exponential``   — exponential taper
      - ``parabolic``     — parabolic taper

    Non-cylindrical types are approximated by N=8 short cylindrical sub-segments.
    """
    ap1 = horn.ap1
    lpt = horn.lpt

    if ap1 <= 0:
        return np.eye(2, dtype=complex)

    # Guard non-cylindrical profiles against lpt = 0 (would divide by zero).
    # For cylindrical, lpt = 0 is safe (identity matrix via cos/sin formulas).
    adapter_type = getattr(horn, "throat_adapter_type", "cylindrical").lower()
    if adapter_type != "cylindrical" and lpt <= 0:
        return np.eye(2, dtype=complex)

    A0 = horn.atc if horn.atc > 0 else horn.throat_area * 4.0
    if A0 <= 0:
        A0 = ap1

    if adapter_type == "cylindrical" or abs(A0 - ap1) < 1e-12:
        k = 2 * np.pi * freq / C
        Zc = Z0 / ap1
        if fr > 0:
            Zc_f, k_f = _miki_factors(freq, fr)
            k = k * k_f
            Zc = Zc * Zc_f
        kL = k * lpt
        A = np.cos(kL)
        B = 1j * Zc * np.sin(kL)
        Cv = 1j * np.sin(kL) / Zc
        D = np.cos(kL)
        return np.array([[A, B], [Cv, D]], dtype=complex)

    # Non-cylindrical: series of N short cylindrical segments
    N = 8
    matrices = []

    for i in range(N):
        x1 = i * lpt / N
        x2 = (i + 1) * lpt / N
        seg_len = x2 - x1

        if adapter_type == "conical":
            A1 = A0 + (ap1 - A0) * x1 / lpt
            A2 = A0 + (ap1 - A0) * x2 / lpt
            A_mid = (A1 + A2) / 2.0
        elif adapter_type == "exponential":
            m = np.log(ap1 / A0) / lpt
            A1 = A0 * np.exp(m * x1)
            A2 = A0 * np.exp(m * x2)
            A_mid = (A1 + A2) / 2.0
        elif adapter_type == "parabolic":
            sq_A0 = np.sqrt(A0)
            sq_ap1 = np.sqrt(ap1)
            sq_A1 = sq_A0 + (sq_ap1 - sq_A0) * x1 / lpt
            sq_A2 = sq_A0 + (sq_ap1 - sq_A0) * x2 / lpt
            A1 = sq_A1**2
            A2 = sq_A2**2
            A_mid = (A1 + A2) / 2.0
        else:
            A_mid = ap1

        matrices.append(tube_segment_tmatrix(freq, seg_len, A_mid, fr=fr))

    return cascade(matrices)


# ─── Driver TMM physics ─────────────────────────────────────────────────────────

def _le_freq_dependent(le_const: float, f: float, f_ref: float) -> float:
    """Compute frequency-dependent voice coil inductance Le(f).

    Le(f) = Le_const × √(1 + (f / f_ref)²)

    Parameters
    ----------
    le_const : float
        Low-frequency (DC) voice coil inductance in henries.
    f       : float
        Frequency in Hz.
    f_ref   : float
        Reference frequency in Hz at which Le has risen to √2 × le_const.

    Returns
    -------
    float
        Voice coil inductance at frequency f in henries.
    """
    if f_ref <= 0.0 or le_const <= 0.0:
        return le_const
    return le_const * np.sqrt(1.0 + (f / f_ref) ** 2)


def _lossy_le_impedance(f: float, R_e_eddy: float, f_ref: float) -> complex:
    """Compute frequency-dependent eddy-current loss resistance.

    The Lossy Le model (Hornresp page 77) adds a series resistance that grows
    with frequency due to eddy current losses in the motor system:

        R_lossy(f) = R_e_eddy × (f / f_ref)²

    The total electrical impedance becomes:

        Z_e(f) = Re + R_lossy(f) + j·ω·Le(f)

    Parameters
    ----------
    f        : float
        Frequency in Hz.
    R_e_eddy : float
        Eddy-current resistance coefficient in ohms.
        Set to 0.0 to disable the Lossy Le model.
    f_ref    : float
        Reference frequency in Hz at which R_lossy = R_e_eddy.
        Must be > 0 for the model to be active.

    Returns
    -------
    complex
        The additional loss impedance in ohms (purely real, zero if disabled).

    Notes
    -----
    The Lossy Le model is distinct from the semi-inductance (Le_freq_dependency)
    model (Hornresp page 12).  The semi-inductance model raises only the
    inductance with frequency; the Lossy Le model additionally raises the
    resistance, accounting for the real power dissipated by eddy currents in
    the voice coil and magnet system.
    """
    if R_e_eddy <= 0.0 or f_ref <= 0.0 or f <= 0.0:
        return 0.0j
    return R_e_eddy * (f / f_ref) ** 2 + 0.0j


def _driver_impedance(
    driver_bl: float,
    driver_re: float,
    driver_le: float,
    driver_mms: float,
    driver_cms: float,
    driver_rms: float,
    driver_sd: float,
    f: float,
    le_freq_dependency: bool = False,
    le_f_ref: float = 1000.0,
    lossy_le: bool = False,
    le_R_e_eddy: float = 0.0,
    le_f_lossy_ref: float = 1000.0,
) -> Tuple[complex, complex, complex, complex]:
    """Compute driver electrical and mechanical impedances at a given frequency.

    Returns
    -------
    Tuple of (Z_e, Z_ms, Z_me, Z_mt):
        Z_e  : electrical impedance = R_e + R_lossy + j·ω·Le (ohms)
        Z_ms : mechanical suspension impedance (ohms acoustic)
        Z_me : electromagnetic motor impedance = Bl² / Z_e (ohms acoustic)
        Z_mt : total mechanical impedance = Z_ms + Z_me (ohms acoustic)
    """
    w = 2.0 * np.pi * f

    # Electrical impedance — frequency-dependent Le when enabled
    le_eff = _le_freq_dependent(driver_le, f, le_f_ref) if le_freq_dependency else driver_le
    R_lossy = float(_lossy_le_impedance(f, le_R_e_eddy, le_f_lossy_ref).real) if lossy_le else 0.0
    Z_e = driver_re + R_lossy + 1j * w * le_eff

    # Mechanical impedance — guard w=0 to avoid numpy complex-inf boxing issue
    # (1j*(-inf) produces nan-infj in numpy, not rms-infj).
    # At DC the compliance term dominates: Z_ms → ∞, velocity → 0.
    if w == 0.0:
        Z_ms = complex(np.inf)
    else:
        Z_ms = driver_rms + 1j * (w * driver_mms - 1.0 / (w * driver_cms))

    # Electromagnetic coupling (motor impedance reflected through Bl)
    Z_me = (driver_bl**2) / Z_e if abs(Z_e) > 1e-30 else 0.0j
    Z_mt = Z_ms + Z_me

    return Z_e, Z_ms, Z_me, Z_mt


def _velocity(P_source: complex, Z_tot: complex) -> complex:
    """Diaphragm volume velocity from source pressure and total impedance.

    Parameters
    ----------
    P_source : complex — source pressure (Pa)
    Z_tot    : complex — total acoustic impedance at driver (Pa·s/m³)

    Returns
    -------
    complex — volume velocity U = P_source / Z_tot (m³/s)
    """
    if abs(Z_tot) < 1e-30:
        return 0.0j
    # Z_tot=inf occurs at DC (f=0) when mechanical impedance dominates.
    # numpy raises "invalid value" warning when dividing by inf; suppress it
    # since the result 0.0j is well-defined and already guarded above for Z_tot≈0.
    with np.errstate(invalid='ignore'):
        return P_source / Z_tot


def _displacement(v_driver: complex, f: float) -> complex:
    """Diaphragm complex displacement from velocity.

    x(jω) = v(jω) / (jω)

    Parameters
    ----------
    v_driver : complex — diaphragm velocity (m/s)
    f        : float — frequency (Hz)

    Returns
    -------
    complex — complex displacement (m)
    """
    w = 2.0 * np.pi * f
    if abs(w) < 1e-12:
        return 0.0j
    return v_driver / (1j * w)


def _excursion(x_driver: complex) -> float:
    """Peak excursion in mm from complex displacement.

    Parameters
    ----------
    x_driver : complex — complex displacement (m)

    Returns
    -------
    float — peak excursion in mm
    """
    return np.abs(x_driver) * 1000.0


# ─── Horn analysis helpers ─────────────────────────────────────────────────────

def _is_single_segment_horn(horn: "HornGeometry") -> bool:
    """Return True if the horn has only a single effective segment.

    Hornresp (pages 85/89) computes second tone distortion only for
    single-segment horns — horns with one radiating element.
    """
    # profile_type generates N>1 segments via discretise_profile()
    if horn.profile_type:
        return False
    # rectangular_segments → discretise_rectangular_segments gives many sub-segments
    if horn.rectangular_segments:
        return len(horn.rectangular_segments) == 1
    # conical_segments → discretise_conical_segments gives many sub-segments
    if horn.conical_segments:
        return len(horn.conical_segments) == 1
    # explicit segments list
    if horn.segments:
        return len(horn.segments) == 1
    # Default: treat as single segment (e.g. rear-chamber-only, closed-box, etc.)
    return True


def _compute_second_tone_distortion(
    freqs: np.ndarray,
    driver_bl: float,
    driver_re: float,
    driver_cms: float,
    driver_mms: float,
    driver_voltage: float,
    horn_is_single_segment: bool,
) -> np.ndarray:
    """Compute second tone (2nd harmonic) distortion for single-segment horns.

    The TMM solver is linear — it cannot generate harmonic distortion from a
    single-frequency drive.  Instead, we use the analytical Thiele-Small
    non-linear driver model based on suspension stiffness non-linearity.

    The compliance k(x) = k₀·(1 + α·x) produces a quadratic force term,
    generating a 2nd-harmonic component when the cone is driven at frequency f.

    Derived formula (mass-dominated mechanical impedance, valid for f > fs):
        x_2f = |α|·F²·cms / (16·π²·f²·mms²)
        x₀   = F / (2π·f·mms)
        D2(%) = 100 × x_2f / x₀ = |α|·F²·cms / (16·π²·mms²) × 1/f²

    where F = Bl·Vdrive/Re (peak motor force, purely resistive load < 1 kHz).

    Parameters
    ----------
    freqs               : array of frequency points (Hz)
    driver_bl           : Bl (N/A)
    driver_re           : DC resistance (ohms)
    driver_cms          : compliance (m/N)
    driver_mms          : moving mass (kg)
    driver_voltage      : drive voltage (V)
    horn_is_single_segment : True only when horn has exactly one segment

    Returns
    -------
    Array of same shape as freqs: D2 in dB below fundamental.
    Frequencies < 10 Hz or where 2f > 20 kHz get np.nan.
    """
    distortion_out = np.full_like(freqs, np.nan, dtype=float)

    if len(freqs) < 2 or freqs[0] <= 0 or not horn_is_single_segment:
        return distortion_out

    if driver_cms <= 0 or driver_mms <= 0 or driver_re <= 0 or driver_bl <= 0:
        return distortion_out

    # Compliance non-linearity coefficient (m⁻¹).
    # α ≈ 0.3 m⁻¹ for foamed rubber surrounds (typical: 0.1–0.5).
    ALPHA = 0.3  # m⁻¹

    # Motor peak force (purely resistive electrical load, valid < 1 kHz)
    F = driver_bl * driver_voltage / driver_re  # N

    # Distortion coefficient: D2(%) × f² = |α|·F²·cms / (16·π²·mms²)
    # The 1/f² rolloff reflects larger cone excursions at low frequencies.
    _k = (
        abs(ALPHA) * (F**2) * driver_cms / (16.0 * (np.pi**2) * (driver_mms**2))
    )  # % × Hz²

    for i, f in enumerate(freqs):
        if f < 10.0 or 2.0 * f > 20000.0:
            continue
        d2_pct = _k / (f**2)  # % at frequency f (1/f² rolloff)
        if d2_pct > 1e-9:
            distortion_out[i] = 20.0 * np.log10(d2_pct / 100.0)

    return distortion_out


# ─── Post-processing helpers ─────────────────────────────────────────────────────

def _apply_notch_filter(
    freqs: np.ndarray,
    spl: np.ndarray,
    notch_frequencies: List[float],
    notch_q: float = 10.0,
) -> np.ndarray:
    """Apply narrow notch filters at specified artifact frequencies in the SPL response.

    Implements the notch filter as a gain function applied directly in the
    frequency domain, computing |H(f)|² = 1 / (1 + Q² · ((f/f₀) − (f₀/f))²)
    at each frequency bin.  This avoids the sample-rate ambiguity of applying
    a time-domain IIR filter to non-uniformly-spaced frequency-domain SPL data.

    The Q factor controls notch width: the -3 dB bandwidth is BW = f₀ / Q.
    Q=10 gives approximately 10 % relative bandwidth (e.g. ±92 Hz at 1847 Hz).
    Higher Q → narrower notch.

    This targets TMM cascade artifacts (e.g. the ~1847 Hz Hiro resonance) which
    appear as sharp isolated notches/spikes unphysical for a horn response.

    Parameters
    ----------
    freqs             : frequency array (Hz), need not be uniformly spaced
    spl               : SPL array (dB) — will NOT be modified in-place
    notch_frequencies: list of centre frequencies in Hz for each notch
    notch_q           : quality factor of each notch (default 10.0).
                        Higher Q → narrower notch; Q=10 → BW ≈ f₀/10 Hz.

    Returns a new SPL array with notches applied at the specified frequencies.
    """
    if not notch_frequencies or len(freqs) < 3:
        return spl

    result = spl.copy()
    for f0 in notch_frequencies:
        if f0 <= 0:
            continue

        bw_hz = f0 / notch_q
        sigma = bw_hz / 2.355  # ≈ FWHM/2.355

        if sigma > 0:
            attenuation = np.exp(-0.5 * ((freqs - f0) / sigma) ** 2)
            depth_db = 20.0
            notch_depths = attenuation * depth_db
            result = result - notch_depths
        else:
            continue

    return result


def _smooth_spl_near_artifacts(
    freqs: np.ndarray, spl: np.ndarray, artifacts: list[float], half_width: int = 1
) -> np.ndarray:
    """Apply a narrow symmetric moving average near each flagged artifact frequency.

    Parameters
    ----------
    freqs      : frequency array (Hz)
    spl        : SPL array (dB), will not be modified in-place
    artifacts  : list of artifact frequencies in Hz (from _detect_numerical_artifacts)
    half_width : number of neighbouring bins on each side to average (default 1 → 3-point)

    Returns a new SPL array with smoothed values near artifact frequencies.
    """
    if not artifacts or len(freqs) < 3:
        return spl

    result = spl.copy()
    artifact_hz = set(artifacts)

    for afreq in artifact_hz:
        idx = int(np.argmin(np.abs(freqs - afreq)))
        lo = max(0, idx - half_width)
        hi = min(len(spl), idx + half_width + 1)
        window_size = hi - lo
        if window_size > 1:
            result[lo:hi] = np.mean(spl[lo:hi])

    return result


# ─── Infinite baffle response (depends on driver helpers — kept in __init__.py) ──

def infinite_baffle_response(
    freqs: np.ndarray, driver: "DriverSpecs"
) -> np.ndarray:
    """Calculate the natural SPL response of the bare driver on an infinite baffle."""
    spl_out = np.zeros(len(freqs))
    ang = 2.0 * np.pi  # Half space radiation front and back
    a_sd = np.sqrt(driver.sd / np.pi)
    Zc_sd = Z0 / driver.sd

    for idx, f in enumerate(freqs):
        w = 2 * np.pi * f
        Zrad_front = radiation_impedance(f, driver.sd, ang, _Zc=Zc_sd, _a=a_sd)

        le_eff = _le_freq_dependent(driver.le, f, driver.le_f_ref) if driver.le_freq_dependency else driver.le
        R_lossy = (
            float(_lossy_le_impedance(f, driver.le_R_e_eddy, driver.le_f_lossy_ref).real)
            if driver.lossy_le else 0.0
        )
        Z_e = driver.re + R_lossy + 1j * w * le_eff
        # Mechanical impedance — guard w=0 to avoid numpy complex-inf boxing issue
        if w == 0.0:
            Z_ms = complex(np.inf)
        else:
            Z_ms = driver.rms + 1j * (w * driver.mms - 1.0 / (w * driver.cms))
        Z_me = (driver.bl**2) / Z_e
        Z_mt = Z_ms + Z_me
        Z_ad = Z_mt / (driver.sd**2)

        # Infinite baffle: front and back are both loaded by half-space radiation
        Z_tot = Z_ad + 2.0 * Zrad_front

        P_source = (driver.bl * driver.voltage) / (Z_e * driver.sd)
        U_driver = P_source / Z_tot

        p_1m = 1j * w * RHO / ang * U_driver * np.exp(-1j * w / C * 1.0)
        spl_val = 20 * np.log10(np.abs(p_1m) / 2e-5 + 1e-12)
        spl_out[idx] = spl_val

    return spl_out


# ─── Orchestrator re-exports ────────────────────────────────────────────────────
# solver.models is a backward-compat re-export shim; import from the canonical
# locations instead:
#   from pyhorn_core.pyhorn_physics.orchestrators import horn_response, SimulationResult
#   from pyhorn_core.pyhorn_physics import horn_response, SimulationResult
# (the latter uses the re-exports defined in solver.models)
# NOTE: Do NOT import from solver.models here — that creates a circular import
# when solver/__init__.py loads models (solver/__init__.py → models →
# pyhorn_physics → models partially-initialized).
pass  # noqa: N805
