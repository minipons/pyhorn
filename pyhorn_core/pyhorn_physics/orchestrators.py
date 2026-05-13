"""
pyhorn_physics.orchestrators — horn acoustic response solvers.

High-level TMM-based horn acoustic response functions: the main BLH solver,
tapped-horn (TH) solver, and compound-horn (CH) solver.

These functions were migrated from solver/models.py as part of the
pyhorn_physics/ decomposition (solver/models.py is now a re-export shim).

Signal Chain (BLH mode, enclosure_type = "BLH")
==============================================
The driver sits at the throat of the horn.  Its front baffle radiates
directly into half-space (direct path); its rear volume drives the horn
path.  Both contributions are summed at the observation point (1 m).

::

    Driver cone
       │
       ├── Z_front_load = Z_rad_front  (half-space radiation)
       │
       └── Z_rear_load = Z_throat + Z_ab  (horn path + rear chamber)
              │
              └── TMM cascade (throat → mouth)
                     │
                     └── Z_rad_mouth  (mouth radiation)

    p_1m = p_direct + p_horn + p_port

For finite transmission-line mode (``enclosure_type = "TL"``) the direct
radiation is excluded from the summed SPL; only the mouth contributes.

For non-BLH/non-TL modes (e.g. sealed, infinite baffle) the driver face
loads into the horn path only (front = throat, rear = chamber).

References
==========
Hornresp manual — relevant pages:

  ===== =====================================================
  Page  Topic
  ===== =====================================================
   12   Semi-inductance Le(f) model
   48–49 Compound Horn (CH) mode
   57–58 Tapped Horn (TH / TH1) mode
   62–65 Rear chamber types: sealed, vented, coupling, SlavBas
   77    FDD directivity model; Lossy Le model
   85/89 Second tone (2nd harmonic) distortion
   92    FDD model
   98    Thermal power compression (voice coil heating)
   113   Futtrup audible group delay limit
  ===== =====================================================

Public API
==========
.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Function / Class
     - Description
   * - ``SimulationResult``
     - Dataclass holding all simulation outputs: SPL, impedance, excursion,
       group delay, on/off-axis response, distortion, thermal compression,
       acoustic power, etc.
   * - ``horn_response(freqs, driver, horn[, T_voice, compute_distortion, …])``
     - Primary BLH / FLH / sealed-box solver.  Runs the full TMM cascade,
       computes throat impedance, driver excursion, summed SPL at 1 m.
       Optional: thermal compression (second pass at T_voice),
       second-tone distortion, FDD directivity, notch filtering of TMM artifacts.
   * - ``horn_response_tapped(freqs, driver, th_geom[, compute_distortion, …])``
     - Tapped Horn (TH1) solver.  The driver is at an interior tap point;
       front loads into the horn proper (throat→mouth), rear loads into
       a rear chamber or infinite baffle.
   * - ``horn_response_compound(freqs, driver, horn, compound[, …])``
     - Compound Horn (CH) solver.  Driver is sandwiched between the main
       horn (front side) and a coupling/rear chamber (rear side).  Supports
       a second driver on the rear side (ch_dual_driver).
   * - ``infinite_baffle_response(freqs, driver)``
     - Natural SPL of the bare driver on an infinite baffle (both sides half-space).
   * - ``compute_thermal_power_compression(freqs, driver, horn, T_voice)``
     - SPL reduction in dB due to voice coil heating.  Runs two solver passes
       (cold Re → hot Re) and compares electrical input power.
   * - ``acoustic_power_to_spl_dB_W_m(p_acoustic_W, sensitivity_db)``
     - Convert acoustic power (W) to SPL in dB/W/m, applying a sensitivity
       offset.  Used for CRIT-3 Hornresp calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
from scipy.special import jv

# ─── Config types ──────────────────────────────────────────────────────────────
if TYPE_CHECKING:
    from pyhorn_core.config.models import (
        DriverSpecs,
        HornGeometry,
        TappedHornGeometry,
        CompoundChamber,
    )

# ─── Solver profiles (segment discretisation) ──────────────────────────────────
from pyhorn_core.solver.profiles import (
    discretise_profile,
    discretise_conical_segments,
    discretise_rectangular_segments,
)

# ─── Physics primitives (already in pyhorn_physics/__init__.py) ─────────────────
from pyhorn_core.pyhorn_physics import (
    RHO,
    C,
    Z0,
    _miki_factors,
    radiation_impedance,
    tube_segment_tmatrix,
    area_step_tmatrix,
    bend_tmatrix,
    compliance_tmatrix,
    rear_chamber_impedance,
    vented_box_impedance,
    passive_radiator_impedance,
    slavbas_impedance,
    transmission_line_impedance,
    throat_adapter_tmatrix,
    cascade,
    _merge_small_throat_segments,
)

# ─── Driver TMM physics ────────────────────────────────────────────────────────
from pyhorn_core.pyhorn_physics import (
    _le_freq_dependent,
    _lossy_le_impedance,
    _driver_impedance,
    _velocity,
    _displacement,
    _excursion,
)

# ─── Distortion + FDD model ────────────────────────────────────────────────────
from pyhorn_core.pyhorn_physics import (
    _is_single_segment_horn,
    _compute_second_tone_distortion as _distortion_physics,
    _fdd_directivity_index,
    _fdd_off_axis_spl,
    _fdd_radiation_angle,
)

# ─── Post-processing helpers ──────────────────────────────────────────────────
from pyhorn_core.pyhorn_physics import (
    _apply_notch_filter,
    _smooth_spl_near_artifacts,
    infinite_baffle_response,
)


# ─────────────────────────────────────────────────────────────────────────────
# SimulationResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SimulationResult:
    freqs: np.ndarray
    spl: np.ndarray
    impedance: np.ndarray
    excursion: np.ndarray
    segments: Optional[List[Tuple[float, ...]]] = None
    ib_spl: Optional[np.ndarray] = None
    direct_spl: Optional[np.ndarray] = None
    horn_spl: Optional[np.ndarray] = None
    group_delay: Optional[np.ndarray] = None
    group_delay_per_period: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    throat_impedance: Optional[np.ndarray] = None
    impedance_phase_deg: Optional[np.ndarray] = None
    segment_widths: Optional[List[float]] = None
    numerical_artifacts: Optional[List[float]] = None
    efficiency_pct: Optional[np.ndarray] = None
    electrical_input_power: Optional[np.ndarray] = None
    off_axis_spl: Optional[np.ndarray] = None
    off_axis_angles: Optional[np.ndarray] = None
    radiation_angle: Optional[float] = None
    fdd_enabled: bool = False
    fdd_di: Optional[np.ndarray] = None
    direction_index: Optional[np.ndarray] = None
    finite_horn_charged: bool = False
    second_tone_distortion: Optional[np.ndarray] = None
    thermal_compression_db: Optional[np.ndarray] = None
    spl_notched: Optional[np.ndarray] = None
    room_gain_db: Optional[np.ndarray] = None
    room_type: Optional[str] = None
    cone_velocity: Optional[np.ndarray] = None
    cone_acceleration: Optional[np.ndarray] = None
    diaphragm_pressure_total: Optional[np.ndarray] = None
    diaphragm_pressure_horn_side: Optional[np.ndarray] = None
    diaphragm_pressure_direct_side: Optional[np.ndarray] = None
    particle_velocity_throat: Optional[np.ndarray] = None
    particle_velocity_mouth: Optional[np.ndarray] = None
    particle_velocity_port: Optional[np.ndarray] = None
    futtrup_gdlimit: Optional[np.ndarray] = None
    acoustic_power: Optional[np.ndarray] = None  # R_rad_mouth * U_mouth_rms² (watts)
    # Acoustic-power-based SPL using dB/W/m reference (Hornresp convention).
    # SPL = 10*log10(P_acoustic / 1e-12) + sensitivity_db dB.
    # This is calibrated to Hornresp's dB/W/m reference — use this instead of
    # `spl` when comparing against Hornresp (CRIT-3 fix).  Set driver.sensitivity_db
    # to the HF calibration offset (e.g. -15.0 dB) to match Hornresp at V=2.83.
    spl_power_based: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def _pressure_to_spl(pressure: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.abs(pressure) / 2e-5 + 1e-12)


def acoustic_power_to_spl_dB_W_m(
    p_acoustic_W: float | np.ndarray,
    sensitivity_db: float = 0.0,
) -> float | np.ndarray:
    """
    Convert acoustic power (watts) to SPL in dB/W/m (dB re 1 W electrical input at 1 m).

    This applies the Hornresp dB/W/m sensitivity reference: acoustic power is
    expressed relative to 1 W electrical input at 1 m distance, using the
    standard acoustic reference (10⁻¹² W).

    The dB/W/m formula:
        SPL(dB/W/m) = 10*log10(P_acoustic / 1e-12) + sensitivity_db

    The sensitivity_db parameter allows calibration to a reference SPL measurement
    (e.g., Hornresp's dB/W/m reference, or a factory-specified driver sensitivity
    like 89 dB/W/m for the FE166NV2 at 1 kHz).

    Parameters
    ----------
    p_acoustic_W
        Acoustic power in watts (from TMM mouth radiation: R_rad * U_rms²).
    sensitivity_db
        Sensitivity offset in dB. Positive → louder reference level.
        For FE166NV2 Hornresp match: sensitivity_db ≈ -8 to -16 dB (HF band,
        frequency-dependent — see CRIT-3 calibration note in BACKLOG.md).

    Returns
    -------
    SPL in dB/W/m.
    """
    P_REF = 1e-12  # acoustic reference power (W)
    with np.errstate(divide="ignore", invalid="ignore"):
        spl = 10.0 * np.log10(np.maximum(np.asarray(p_acoustic_W), P_REF) / P_REF)
        spl += sensitivity_db
    return spl


def _detect_numerical_artifacts(freqs: np.ndarray, spl: np.ndarray) -> List[float]:
    """Detect TMM numerical artifacts — unphysically large SPL spikes/dips."""
    if len(freqs) < 5:
        return []

    dspl = np.diff(spl)
    artifacts_set: set[float] = set()
    THRESHOLD = 20.0
    window = 5

    for i in range(1, len(dspl)):
        d = dspl[i - 1]
        next_d = dspl[i] if i < len(dspl) else 0.0

        is_isolated_spike = abs(d) > THRESHOLD and d * next_d < 0

        lo = max(0, i - window)
        hi = min(len(spl), i + window + 1)
        local_median = float(np.median(spl[lo:hi]))
        deviation = abs(spl[i] - local_median)
        is_trend_break = deviation > THRESHOLD

        if is_isolated_spike or is_trend_break:
            artifacts_set.add(float(freqs[i]))

    return sorted(artifacts_set)


def _compute_second_tone_distortion(
    freqs: np.ndarray,
    driver: "DriverSpecs",
    horn: "HornGeometry",
    spl_fundamental: np.ndarray,
) -> np.ndarray:
    """Backward-compat shim — calls pyhorn_physics._compute_second_tone_distortion."""
    return _distortion_physics(
        freqs,
        driver.bl,
        driver.re,
        driver.cms,
        driver.mms,
        driver.voltage,
        _is_single_segment_horn(horn),
    )


def _compute_futtrup_gdlimit(freqs: np.ndarray) -> np.ndarray:
    """Futtrup audible group delay limit (Hornresp page 113).

    GDlimit = 1000 × 1160.6 / (5643 × f^0.81511 − f)  [ms]
    Below ~50 Hz the denominator approaches zero; clamp to a safe upper bound.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = 5643.0 * freqs**0.81511 - freqs
        return np.where(
            denominator > 1.0,
            1000.0 * 1160.6 / denominator,
            1000.0 * 1160.6 / 1.0,
        )


def compute_thermal_power_compression(
    freqs: np.ndarray,
    driver: "DriverSpecs",
    horn: "HornGeometry",
    T_voice: float = 100.0,
) -> np.ndarray:
    """Compute thermal power compression: SPL reduction (dB) due to voice coil heating."""
    if T_voice <= 20.0:
        return np.zeros_like(freqs, dtype=float)

    alpha = getattr(driver, "alpha_re", 0.00393)
    re_nominal = driver.re
    re_heated = re_nominal * (1.0 + alpha * (T_voice - 20.0))

    result_nominal = _run_horn_response_internal(freqs, driver, horn)

    from dataclasses import replace
    driver_hot = replace(driver, re=re_heated)
    result_hot = _run_horn_response_internal(freqs, driver_hot, horn)

    p_elec_nom = np.zeros(len(freqs))
    p_elec_hot = np.zeros(len(freqs))

    for idx, f in enumerate(freqs):
        w = 2 * np.pi * f
        z_in_nom = result_nominal.impedance[idx]
        z_in_hot = result_hot.impedance[idx]
        le_eff = _le_freq_dependent(driver.le, f, driver.le_f_ref) if driver.le_freq_dependency else driver.le
        R_lossy = (
            float(_lossy_le_impedance(f, driver.le_R_e_eddy, driver.le_f_lossy_ref).real)
            if driver.lossy_le else 0.0
        )
        z_e_sq_nom = (re_nominal + R_lossy)**2 + (w * le_eff) ** 2
        z_e_sq_hot = (re_heated + R_lossy)**2 + (w * le_eff) ** 2

        if z_e_sq_nom > 0:
            p_elec_nom[idx] = driver.voltage**2 * z_in_nom.real / z_e_sq_nom
        if z_e_sq_hot > 0:
            p_elec_hot[idx] = driver.voltage**2 * z_in_hot.real / z_e_sq_hot

    p_elec_nom = np.maximum(p_elec_nom, 1e-30)
    p_elec_hot = np.maximum(p_elec_hot, 1e-30)

    ratio = p_elec_hot / p_elec_nom
    compression_db = 10.0 * np.log10(ratio)
    compression_db = np.clip(compression_db, None, 0.0)

    return compression_db


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrators
# ─────────────────────────────────────────────────────────────────────────────

def _run_horn_response_internal(
    freqs: np.ndarray, driver: "DriverSpecs", horn: "HornGeometry"
) -> SimulationResult:
    """Run the solver without triggering thermal compression computation."""
    return _horn_response_impl(freqs, driver, horn, _thermal_T_voice=None)


def horn_response(
    freqs: np.ndarray,
    driver: "DriverSpecs",
    horn: "HornGeometry",
    T_voice: Optional[float] = None,
    compute_distortion: bool = True,
    off_axis_angles: Optional[np.ndarray] = None,
    notch_filter: bool = False,
    notch_frequencies: Optional[List[float]] = None,
    notch_q: float = 10.0,
    fdd_mode: bool = False,
    fdd_fc: float = 300.0,
    fdd_dmax: float = 5.0,
) -> SimulationResult:
    """
    Compute acoustic pressure, electrical impedance, and driver excursion.

    Parameters
    ----------
    T_voice : float, optional
        Voice coil temperature in °C for thermal power compression modeling.
        When provided (> 20°C), a second solver pass at 20°C is run and the
        SPL reduction due to voice coil heating is attached as thermal_compression_db.
        Reference: Hornresp page 98.
    compute_distortion : bool, default True
        When True, compute second tone distortion for single-segment horns.
        Disable to save CPU time on large frequency sweeps.
    off_axis_angles : np.ndarray, optional
        Array of off-axis angles in degrees for directivity computation.
        Defaults to [0, 15, 30, 45, 60, 75, 90] if not provided.
    notch_filter : bool, default False
        When True, apply narrow IIR notch filters at the specified artifact
        frequencies to suppress TMM numerical artifacts (e.g. ~1847 Hz Hiro resonance).
        The notched SPL is stored in result.spl_notched alongside the raw SPL.
    notch_frequencies : list of float, optional
        Centre frequencies in Hz for notch filters.  Required when notch_filter=True.
        Defaults to [1847, 2508, 2732, 2852, 2969] Hz (Hiro project characterisation).
    notch_q : float, default 10.0
        Quality factor for each notch filter.  Higher Q → narrower notch.
        Q=10 gives approximately 10 % bandwidth at the -3 dB points.
    fdd_mode : bool, default False
        When True, use the FDD (Frequency Dependent Directivity) model instead
        of the standard piston (Levine/Inglis) model for off-axis directivity.
        Reference: Hornresp pages 77 and 92 (FDD model).
    fdd_fc : float, default 300.0
        FDD characteristic transition frequency in Hz.
    fdd_dmax : float, default 5.0
        FDD maximum directivity index in dB.
    """
    return _horn_response_impl(
        freqs, driver, horn, _thermal_T_voice=T_voice,
        compute_distortion=compute_distortion, off_axis_angles=off_axis_angles,
        notch_filter=notch_filter, notch_frequencies=notch_frequencies, notch_q=notch_q,
        fdd_mode=fdd_mode, fdd_fc=fdd_fc, fdd_dmax=fdd_dmax,
    )


def horn_response_tapped(
    freqs: np.ndarray,
    driver: "DriverSpecs",
    th_geom: "TappedHornGeometry",
    compute_distortion: bool = True,
    off_axis_angles: Optional[np.ndarray] = None,
) -> SimulationResult:
    """Compute the acoustic response of a Tapped Horn (TH / TH1 mode).

    The driver is positioned at an *interior* point of the horn (S2 or S3).
    The front of the driver loads into the horn proper (tap → mouth).
    The rear of the driver loads into a rear chamber or free space.

    Reference: Hornresp manual pages 057–058.
    """
    rho = 1.21
    c = 343.0
    Z0_local = rho * c

    if off_axis_angles is None:
        off_axis_angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])

    n_freqs = len(freqs)
    n_angles = len(off_axis_angles)

    front_segments: list[tuple[float, float]] = []
    for sec in th_geom.front_sections:
        n_seg = max(th_geom.n_segments, 10)
        segs = discretise_profile(
            sec.profile_type,
            sec.start_area,
            sec.end_area,
            sec.length,
            n_seg,
            hyperbolic_t=sec.hyperbolic_t if sec.hyperbolic_t is not None else 1.0,
        )
        front_segments.extend((dx, A_avg) for dx, A_avg, *_ in segs)

    rear_segments: list[tuple[float, float]] = []
    for sec in th_geom.rear_sections:
        n_seg = max(th_geom.n_segments, 10)
        segs = discretise_profile(
            sec.profile_type,
            sec.start_area,
            sec.end_area,
            sec.length,
            n_seg,
            hyperbolic_t=sec.hyperbolic_t if sec.hyperbolic_t is not None else 1.0,
        )
        rear_segments.extend((dx, A_avg) for dx, A_avg, *_ in segs)

    mouth_area = th_geom.front_sections[-1].end_area if th_geom.front_sections else 0.0
    mouth_radius = np.sqrt(mouth_area / np.pi)
    S_d = driver.sd

    def rear_load_at_tap(f: float) -> complex:
        omega = 2.0 * np.pi * f
        if th_geom.rear_load_type == "rear_chamber" and th_geom.rear_chamber:
            rc = th_geom.rear_chamber
            C_rc = rc.vrc / (rho * c**2) if rc.vrc > 0 else 0.0
            L_rc = rho * rc.lrc / S_d if rc.lrc > 0 and S_d > 0 else 0.0
            R_rc = rho * rc.fr_rc / S_d if rc.fr_rc > 0 and S_d > 0 else 0.0
            Z_c = 0.0
            if C_rc > 0:
                Z_c += 1.0 / (1j * omega * C_rc)
            if L_rc > 0:
                Z_c += 1j * omega * L_rc
            return Z_c + R_rc
        elif th_geom.rear_load_type == "infinite_baffle":
            return radiation_impedance(f, mouth_area, np.pi)
        else:
            return radiation_impedance(f, S_d, 2.0 * np.pi)

    spl = np.zeros(n_freqs)
    impedance = np.zeros(n_freqs, dtype=complex)
    cone_excursion = np.zeros(n_freqs)
    cone_velocity_arr = np.zeros(n_freqs)
    cone_acceleration_arr = np.zeros(n_freqs)
    electrical_power = np.zeros(n_freqs)
    acoustic_power = np.zeros(n_freqs)
    efficiency = np.zeros(n_freqs)
    group_delay = np.zeros(n_freqs)
    phase = np.zeros(n_freqs)
    off_axis_spl_arr = np.zeros((n_freqs, n_angles))
    direction_index_arr = np.zeros((n_freqs, n_angles)) if n_angles > 0 else None

    for i, f in enumerate(freqs):
        if f < 1.0:
            f = 1.0

        omega = 2.0 * np.pi * f
        k = omega / c

        Z_rad_mouth = radiation_impedance(f, mouth_area, th_geom.ang)
        Z_rad_mouth_cplx = complex(Z_rad_mouth.real, Z_rad_mouth.imag)

        Z_in_front = Z_rad_mouth_cplx
        for dx, A_avg in reversed(front_segments):
            Zc_seg = Z0_local / A_avg
            kl = k * dx
            if abs(kl) < 1e-12:
                continue
            denom = Zc_seg + 1j * Z_in_front * np.sin(kl)
            if abs(denom) < 1e-30:
                Z_in_front = Zc_seg
            else:
                Z_in_front = Zc_seg * (Z_in_front + 1j * Zc_seg * np.tan(kl)) / (
                    Zc_seg + 1j * Z_in_front * np.tan(kl)
                )

        Z_tap_rear = rear_load_at_tap(f)

        Y_front = 1.0 / Z_in_front if abs(Z_in_front) > 1e-30 else 0.0
        Y_rear = 1.0 / Z_tap_rear if abs(Z_tap_rear) > 1e-30 else 0.0
        Y_total = Y_front + Y_rear
        Z_total_ac = 1.0 / Y_total if abs(Y_total) > 1e-30 else 1e30

        Z_mech_driver = (driver.mms * omega * 1j +
                         driver.rms +
                         1.0 / (driver.cms * 1j * omega) if omega > 0 else 0.0)
        Z_mech_total = Z_mech_driver + Z_total_ac * S_d**2

        Bl = driver.bl
        le_eff = _le_freq_dependent(driver.le, f, driver.le_f_ref) if driver.le_freq_dependency else driver.le
        R_lossy = (
            float(_lossy_le_impedance(f, driver.le_R_e_eddy, driver.le_f_lossy_ref).real)
            if driver.lossy_le else 0.0
        )
        Z_e_local = driver.re + R_lossy + 1j * omega * le_eff
        F_bl = Bl * driver.voltage / Z_e_local
        if abs(Z_mech_total) > 1e-12:
            u_cone = F_bl / Z_mech_total
        else:
            u_cone = 0.0
        U_tap = u_cone * S_d

        U_mouth = U_tap
        p_tap = U_tap / Y_front if abs(Y_front) > 1e-30 else 0.0

        for dx, A_avg in front_segments:
            Zc_seg = Z0_local / A_avg
            kl = k * dx
            if abs(kl) < 1e-12:
                continue
            cos_kl = np.cos(kl)
            sin_kl = np.sin(kl)
            p_next = cos_kl * p_tap + 1j * Zc_seg * sin_kl * U_mouth
            u_next = (1j / Zc_seg) * sin_kl * p_tap + cos_kl * U_mouth
            p_tap = p_next
            U_mouth = u_next

        Le_val = driver.le
        if getattr(driver, "le_freq_dependency", False):
            f_ref = getattr(driver, "le_f_ref", 1000.0)
            Le_val = _le_freq_dependent(driver.le, f, f_ref)
        Z_e = driver.re + 1j * omega * Le_val
        Z_motor = Z_e + (Bl**2) / Z_mech_total if abs(Z_mech_total) > 1e-12 else Z_e
        impedance[i] = Z_motor

        if omega > 0:
            cone_excursion[i] = abs(u_cone) / omega
            cone_velocity_arr[i] = abs(u_cone)
            cone_acceleration_arr[i] = abs(1j * omega * u_cone)
        else:
            cone_excursion[i] = 0.0
            cone_velocity_arr[i] = 0.0
            cone_acceleration_arr[i] = 0.0

        p_mouth_mag = abs(Z_rad_mouth_cplx * U_mouth) / np.sqrt(2)
        SPL_ref = 20.0 * np.log10(max(p_mouth_mag / 2e-5, 1e-12))
        spl[i] = SPL_ref

        U_rms_sq = (abs(u_cone) / np.sqrt(2)) ** 2 if driver.re > 0 else 0.0
        P_elec = U_rms_sq / driver.re
        electrical_power[i] = P_elec
        P_ac = max(Z_rad_mouth.real, 0.0) * (abs(U_mouth) / np.sqrt(2)) ** 2
        acoustic_power[i] = P_ac
        if P_elec > 1e-12:
            efficiency[i] = min(P_ac / P_elec, 1.0) * 100.0
        else:
            efficiency[i] = 0.0

        p_complex = Z_rad_mouth_cplx * U_mouth
        phase[i] = np.angle(p_complex)
        group_delay[i] = -np.gradient(phase, freqs)[i] if i > 0 else 0.0

        for j, theta_deg in enumerate(off_axis_angles):
            theta = np.radians(theta_deg)
            ka_mouth = k * mouth_radius
            if abs(ka_mouth * np.sin(theta)) < 1e-9:
                dir_factor = 1.0
            else:
                ka_theta = ka_mouth * np.sin(theta)
                dir_factor = 2.0 * (
                    np.sin(ka_theta) - ka_theta * np.cos(ka_theta)
                ) / (ka_theta**3)
                dir_factor = max(dir_factor, 0.0)
            off_axis_spl_arr[i, j] = SPL_ref + 10.0 * np.log10(max(dir_factor, 1e-12))

    spl_out = np.array(spl, dtype=float)

    if direction_index_arr is not None:
        for j in range(n_angles):
            theta = np.radians(off_axis_angles[j])
            ka_mouth_arr = 2.0 * np.pi * freqs * mouth_radius / c
            with np.errstate(divide="ignore", invalid="ignore"):
                ka_theta = np.where(
                    abs(ka_mouth_arr * np.sin(theta)) < 1e-9,
                    1.0,
                    ka_mouth_arr * np.sin(theta),
                )
                dir_factor = np.where(
                    abs(ka_theta) < 1e-9,
                    1.0,
                    2.0 * (np.sin(ka_theta) - ka_theta * np.cos(ka_theta))
                    / (ka_theta**3),
                )
                dir_factor = np.maximum(dir_factor, 1e-12)
            direction_index_arr[:, j] = 10.0 * np.log10(dir_factor)

    return SimulationResult(
        freqs=freqs,
        spl=spl_out,
        impedance=impedance,
        impedance_phase_deg=np.angle(impedance) * 180.0 / np.pi,
        excursion=cone_excursion,
        cone_velocity=cone_velocity_arr,
        cone_acceleration=cone_acceleration_arr,
        group_delay=group_delay,
        group_delay_per_period=group_delay / 1000.0 * freqs if group_delay is not None else None,
        phase=phase,
        efficiency_pct=efficiency,
        electrical_input_power=electrical_power,
        acoustic_power=acoustic_power,
        off_axis_spl=off_axis_spl_arr if n_angles > 0 else None,
        off_axis_angles=off_axis_angles if n_angles > 0 else None,
        direction_index=direction_index_arr,
        radiation_angle=np.nan,
        throat_impedance=None,
        spl_notched=spl_out,
        thermal_compression_db=None,
        second_tone_distortion=None,
        particle_velocity_throat=None,
        particle_velocity_mouth=None,
        particle_velocity_port=None,
        futtrup_gdlimit=_compute_futtrup_gdlimit(freqs),
    )


def horn_response_compound(
    freqs: np.ndarray,
    driver: "DriverSpecs",
    horn: "HornGeometry",
    compound: "CompoundChamber",
    compute_distortion: bool = True,
    off_axis_angles: Optional[np.ndarray] = None,
) -> SimulationResult:
    """Compute the acoustic response of a Compound Horn (CH mode).

    In Hornresp CH mode the driver is sandwiched between two horn structures:
      - **Front side**: the driver cone drives the main horn (S1→S4).
      - **Rear side**: the driver cone also drives the rear coupling chamber.

    Reference: Hornresp manual pages 048–049, 059.
    """
    rho = RHO
    c = C
    Z0_rc = Z0

    if off_axis_angles is None:
        off_axis_angles = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    n_freqs = len(freqs)
    n_angles = len(off_axis_angles)

    if horn.profile_type:
        segs = discretise_profile(
            horn.profile_type,
            horn.throat_area,
            horn.mouth_area,
            horn.path_length,
            horn.n_segments,
            horn.hyperbolic_t,
        )
        main_segments = [(s[0], s[1]) for s in segs]
    elif horn.sections:
        main_segments = []
        for sec in horn.sections:
            n_seg = max(horn.n_segments, 10)
            segs = discretise_profile(
                sec.profile_type,
                sec.start_area,
                sec.end_area,
                sec.length,
                n_seg,
                hyperbolic_t=sec.hyperbolic_t if sec.hyperbolic_t is not None else 1.0,
            )
            main_segments.extend((s[0], s[1]) for s in segs)
    elif horn.segments:
        main_segments = [(s[0], s[1]) for s in horn.segments]
    else:
        main_segments = []

    main_mouth_area = max(main_segments[-1][1] if main_segments else horn.mouth_area, 1e-12)
    secondary_mouth_area = max(compound.secondary_mouth_area, 1e-12)
    main_path = sum(seg[0] for seg in main_segments)

    secondary_segments: List[Tuple[float, float]] = []
    if compound.ch_dual_driver and compound.rear_driver is not None:
        sec_profile = horn.profile_type or "exponential"
        sec_path = compound.lrc_rear
        if sec_path <= 0:
            sec_path = 0.1
        sec_seg_count = max(horn.n_segments, 10)
        from pyhorn_core.solver.profiles import discretise_profile as _disc
        sec_segs = _disc(
            sec_profile,
            main_mouth_area,
            secondary_mouth_area,
            sec_path,
            sec_seg_count,
            hyperbolic_t=horn.hyperbolic_t,
        )
        secondary_segments = [(s[0], s[1]) for s in sec_segs]

    secondary_path = sum(seg[0] for seg in secondary_segments) if secondary_segments else 0.0

    S_d = driver.sd
    a_sd = np.sqrt(S_d / np.pi)
    Zc_sd = Z0_rc / S_d

    ang_main = horn.ang
    ang_rear = compound.secondary_mouth_ang

    spl = np.zeros(n_freqs)
    impedance = np.zeros(n_freqs, dtype=complex)
    excursion = np.zeros(n_freqs)
    cone_velocity_arr = np.zeros(n_freqs)
    cone_accel_arr = np.zeros(n_freqs)
    group_delay = np.zeros(n_freqs)
    phase = np.zeros(n_freqs)
    efficiency = np.zeros(n_freqs)
    electrical_power = np.zeros(n_freqs)
    main_pressure = np.zeros(n_freqs, dtype=complex)
    rear_pressure = np.zeros(n_freqs, dtype=complex)

    for i, f in enumerate(freqs):
        if f < 1.0:
            f = 1.0

        w = 2.0 * np.pi * f
        k = w / c

        Zrad_main = radiation_impedance(f, main_mouth_area, ang_main)

        Z_in_front = complex(Zrad_main.real, Zrad_main.imag)
        for dx, A_avg in reversed(main_segments):
            Zc_seg = Z0_rc / A_avg
            kl = k * dx
            if abs(kl) < 1e-12:
                continue
            tan_kl = np.tan(kl)
            denom = Zc_seg + 1j * Z_in_front * tan_kl
            if abs(denom) < 1e-30:
                Z_in_front = Zc_seg
            else:
                Z_in_front = Zc_seg * (Z_in_front + 1j * Zc_seg * tan_kl) / denom

        C_rc = compound.vrc_rear / (rho * c**2) if compound.vrc_rear > 0 else 0.0
        M_rc = rho * compound.lrc_rear / S_d if compound.lrc_rear > 0 and S_d > 0 else 0.0
        C_nck = compound.vtc_rear / (rho * c**2) if compound.vtc_rear > 0 else 0.0

        Z_rear_chamber = 0.0j
        if C_rc > 0:
            Z_rear_chamber += 1.0 / (1j * w * C_rc)
        if M_rc > 0:
            Z_rear_chamber += 1j * w * M_rc
        if C_nck > 0:
            Z_rear_chamber += 1.0 / (1j * w * C_nck)

        Z_rear_total = Z_rear_chamber
        if compound.secondary_mouth_area > 0 and main_segments:
            Zrad_sec = radiation_impedance(f, secondary_mouth_area, ang_rear)
            Z_rear_total = Z_rear_chamber + Zrad_sec
        elif abs(Z_rear_chamber) < 1e-12:
            Z_rear_total = radiation_impedance(f, S_d, 2.0 * np.pi, _Zc=Zc_sd, _a=a_sd)

        Y_front = 1.0 / Z_in_front if abs(Z_in_front) > 1e-30 else 0.0j
        Y_rear = 1.0 / Z_rear_total if abs(Z_rear_total) > 1e-30 else 0.0j
        Y_total = Y_front + Y_rear
        Z_total_ac = 1.0 / Y_total if abs(Y_total) > 1e-30 else 1e30

        le_eff = _le_freq_dependent(driver.le, f, driver.le_f_ref) if driver.le_freq_dependency else driver.le
        R_lossy = (
            float(_lossy_le_impedance(f, driver.le_R_e_eddy, driver.le_f_lossy_ref).real)
            if driver.lossy_le else 0.0
        )
        Z_e = driver.re + R_lossy + 1j * w * le_eff
        Z_ms = driver.rms + 1j * (w * driver.mms - 1.0 / (w * driver.cms))
        Z_me = (driver.bl**2) / Z_e
        Z_mt = Z_ms + Z_me

        Z_mech_total = Z_mt + Z_total_ac * S_d**2

        Z_mech_total_load = Z_ms + (Z_in_front + Z_rear_total) * S_d**2
        Z_in_elec = Z_e + driver.bl**2 / Z_mech_total_load if abs(Z_mech_total_load) > 1e-12 else Z_e

        Bl = driver.bl
        F_bl = Bl * driver.voltage / Z_e
        if abs(Z_mech_total) > 1e-12:
            u_cone = F_bl / Z_mech_total
        else:
            u_cone = 0.0 + 0.0j
        U_driver = u_cone * S_d
        U_rear_for_power = U_driver

        U_throat = -U_driver
        P_throat = U_throat * Z_in_front if abs(Z_in_front) > 1e-30 else 0.0j

        p_tap = P_throat
        U_mouth_main = U_throat
        for dx, A_avg in main_segments:
            Zc_seg = Z0_rc / A_avg
            kl = k * dx
            if abs(kl) < 1e-12:
                continue
            sin_kl = np.sin(kl)
            cos_kl = np.cos(kl)
            p_next = cos_kl * p_tap + 1j * Zc_seg * sin_kl * U_mouth_main
            u_next = (1j / Zc_seg) * sin_kl * p_tap + cos_kl * U_mouth_main
            p_tap = p_next
            U_mouth_main = u_next

        P_main_mouth = p_tap
        U_main_mouth = U_mouth_main

        r_main = 1.0 + main_path
        p_horn_front = (
            1j * w * rho / ang_main
            * U_main_mouth
            * np.exp(-1j * w / c * r_main)
        )
        main_pressure[i] = p_horn_front

        if compound.ch_dual_driver and compound.rear_driver is not None:
            rd = compound.rear_driver
            S_rd = rd.sd
            Zc_rd = Z0_rc / S_rd

            Zrad_sec = radiation_impedance(f, secondary_mouth_area, ang_rear)
            Z_in_sec = complex(Zrad_sec.real, Zrad_sec.imag)
            for dx, A_avg in reversed(secondary_segments):
                Zc_seg = Z0_rc / A_avg
                kl = k * dx
                if abs(kl) < 1e-12:
                    continue
                tan_kl = np.tan(kl)
                denom = Zc_seg + 1j * Z_in_sec * tan_kl
                if abs(denom) < 1e-30:
                    Z_in_sec = Zc_seg
                else:
                    Z_in_sec = Zc_seg * (Z_in_sec + 1j * Zc_seg * tan_kl) / denom

            le_eff_rd = (
                _le_freq_dependent(rd.le, f, rd.le_f_ref)
                if rd.le_freq_dependency else rd.le
            )
            Z_e_rd = rd.re + 1j * w * le_eff_rd
            Z_ms_rd = rd.rms + 1j * (w * rd.mms - 1.0 / (w * rd.cms))
            Z_me_rd = (rd.bl**2) / Z_e_rd
            Z_mech_rd = Z_ms_rd + Z_me_rd + Z_in_sec * S_rd**2

            Bl_rd = rd.bl
            U_e_rd = rd.voltage / rd.re
            F_bl_rd = Bl_rd * U_e_rd / rd.re
            if abs(Z_mech_rd) > 1e-12:
                u_cone_rd = F_bl_rd / Z_mech_rd
            else:
                u_cone_rd = 0.0 + 0.0j
            U_driver_rd = u_cone_rd * S_rd
            U_rear_for_power = U_driver_rd

            U_sec_in = -U_driver_rd
            P_sec_in = U_sec_in * Z_in_sec if abs(Z_in_sec) > 1e-30 else 0.0j
            p_sec_tap = P_sec_in
            U_sec_mouth = U_sec_in
            for dx, A_avg in secondary_segments:
                Zc_seg = Z0_rc / A_avg
                kl = k * dx
                if abs(kl) < 1e-12:
                    continue
                sin_kl = np.sin(kl)
                cos_kl = np.cos(kl)
                p_next = cos_kl * p_sec_tap + 1j * Zc_seg * sin_kl * U_sec_mouth
                u_next = (1j / Zc_seg) * sin_kl * p_sec_tap + cos_kl * U_sec_mouth
                p_sec_tap = p_next
                U_sec_mouth = u_next

            P_sec_mouth = p_sec_tap

            Zrad_sec_mouth = radiation_impedance(f, secondary_mouth_area, ang_rear)
            r_sec = 1.0 + secondary_path
            p_rear = (
                1j * w * rho / ang_rear
                * U_sec_mouth
                * np.exp(-1j * w / c * r_sec)
            )

            Z_in_elec_rd = (
                Z_e_rd + Bl_rd**2 / Z_mech_rd
                if abs(Z_mech_rd) > 1e-12 else Z_e_rd
            )
            Z_in_elec_total = 1.0 / (
                1.0 / Z_in_elec + 1.0 / Z_in_elec_rd
            ) if abs(Z_in_elec_rd) > 1e-12 else Z_in_elec
            Zrad_rear = Zrad_sec_mouth
            if w > 0:
                cone_velocity_arr[i] = abs(u_cone)
                cone_accel_arr[i] = abs(1j * w * u_cone)
            else:
                cone_velocity_arr[i] = 0.0
                cone_accel_arr[i] = 0.0
        else:
            ang_rear_eff = ang_rear if compound.secondary_mouth_area > 0 else 2.0 * np.pi
            Zrad_rear = radiation_impedance(f, S_d, ang_rear_eff, _Zc=Zc_sd, _a=a_sd)
            r_rear = 1.0
            if abs(Z_rear_total) > 1e-30:
                p_rear = Zrad_rear * U_driver * np.exp(-1j * w / c * r_rear)
            else:
                p_rear = 0.0j
            Z_in_elec_total = Z_in_elec

        rear_pressure[i] = p_rear

        p_1m = p_horn_front + p_rear

        impedance[i] = Z_in_elec_total

        if w > 0:
            excursion[i] = abs(u_cone) / w * 1000.0 * np.sqrt(2)
            if not (compound.ch_dual_driver and compound.rear_driver is not None):
                cone_velocity_arr[i] = abs(u_cone)
                cone_accel_arr[i] = abs(1j * w * u_cone)
        else:
            excursion[i] = 0.0
            if not (compound.ch_dual_driver and compound.rear_driver is not None):
                cone_velocity_arr[i] = 0.0
                cone_accel_arr[i] = 0.0

        if i == 0:
            phase[i] = np.angle(p_1m)
        else:
            phase[i] = np.angle(p_1m)
        group_delay[i] = 0.0

        z_e_sq_mag = driver.re**2 + (w * le_eff) ** 2
        p_elec = driver.voltage**2 * Z_in_elec.real / z_e_sq_mag
        electrical_power[i] = p_elec
        P_ac_main = max(Zrad_main.real, 0.0) * (abs(U_main_mouth) / np.sqrt(2)) ** 2
        P_ac_rear = max(Zrad_rear.real, 0.0) * (abs(U_rear_for_power) / np.sqrt(2)) ** 2
        P_acoustic = P_ac_main + P_ac_rear
        efficiency[i] = (100.0 * P_acoustic / p_elec) if p_elec > 1e-12 else 0.0

        spl[i] = _pressure_to_spl(np.array([p_1m]))[0]

    phase = np.unwrap(np.angle(main_pressure + rear_pressure))
    omega_arr = 2.0 * np.pi * freqs
    if len(freqs) >= 2:
        group_delay = -np.gradient(phase, omega_arr) * 1000.0
    else:
        group_delay = np.zeros_like(freqs, dtype=float)

    off_axis_spl_out = np.zeros((n_freqs, n_angles))
    a_mouth_ch = np.sqrt(main_mouth_area / np.pi)
    ka_arr = 2.0 * np.pi * freqs * a_mouth_ch / c
    for j, ang_deg in enumerate(off_axis_angles):
        ang_rad = np.radians(ang_deg)
        sin_t = np.sin(ang_rad)
        x = ka_arr * sin_t
        x_safe = np.where(x < 0.05, 0.05, x)
        j1_vals = jv(1, x_safe)
        sinc_sq = (2.0 * j1_vals / (x_safe + 1e-12)) ** 2
        D = np.where(ka_arr < 0.05, 1.0, sinc_sq)
        off_axis_spl_out[:, j] = 10.0 * np.log10(D + 1e-12)

    direction_index_out = np.zeros((n_freqs, n_angles), dtype=float)
    for j, ang_deg in enumerate(off_axis_angles):
        ang_rad = np.radians(ang_deg)
        sin_t = np.sin(ang_rad)
        x = ka_arr * sin_t
        x_safe = np.where(x < 0.05, 0.05, x)
        j1_vals = jv(1, x_safe)
        sinc_unsq = 2.0 * j1_vals / (x_safe + 1e-12)
        D_factor = np.where(ka_arr < 0.05, 1.0, sinc_unsq)
        direction_index_out[:, j] = 10.0 * np.log10(np.maximum(D_factor, 1e-12))

    second_tone_dist = (
        _compute_second_tone_distortion(freqs, driver, horn, spl)
        if compute_distortion and len(main_segments) == 1
        else None
    )

    return SimulationResult(
        freqs=freqs,
        spl=spl,
        impedance=impedance,
        impedance_phase_deg=np.angle(impedance) * 180.0 / np.pi,
        excursion=excursion,
        cone_velocity=cone_velocity_arr,
        cone_acceleration=cone_accel_arr,
        group_delay=group_delay,
        group_delay_per_period=group_delay / 1000.0 * freqs if group_delay is not None else None,
        phase=phase,
        efficiency_pct=efficiency,
        electrical_input_power=electrical_power,
        off_axis_spl=off_axis_spl_out if n_angles > 0 else None,
        off_axis_angles=off_axis_angles if n_angles > 0 else None,
        direction_index=direction_index_out,
        radiation_angle=np.nan,
        throat_impedance=None,
        spl_notched=spl,
        thermal_compression_db=None,
        second_tone_distortion=second_tone_dist,
        particle_velocity_throat=None,
        particle_velocity_mouth=None,
        particle_velocity_port=None,
        futtrup_gdlimit=_compute_futtrup_gdlimit(freqs),
    )


def _horn_response_impl(
    freqs: np.ndarray,
    driver: "DriverSpecs",
    horn: "HornGeometry",
    _thermal_T_voice: Optional[float] = None,
    compute_distortion: bool = True,
    off_axis_angles: Optional[np.ndarray] = None,
    notch_filter: bool = False,
    notch_frequencies: Optional[List[float]] = None,
    notch_q: float = 10.0,
    fdd_mode: bool = False,
    fdd_fc: float = 300.0,
    fdd_dmax: float = 5.0,
) -> SimulationResult:
    """Internal implementation of the horn response solver."""

    enc = horn.enclosure_type.upper()
    is_blh = enc == "BLH"
    is_tl = enc == "TL"  # Finite transmission line: mouth radiation only, no direct summing

    path_diff = horn.path_diff
    if is_blh and path_diff == 0.0:
        if horn.rectangular_segments:
            path_diff = sum(seg[4] for seg in horn.rectangular_segments)
        elif horn.conical_segments:
            path_diff = sum(seg[2] for seg in horn.conical_segments)

    bends_ext = None
    bend_positions = None
    segment_widths = None
    lem_step_model = horn.lem_step_model
    lem_step_strength = horn.lem_step_strength
    lem_step_resistance = horn.lem_step_resistance
    if horn.profile_type:
        segments = discretise_profile(
            horn.profile_type,
            horn.throat_area,
            horn.mouth_area,
            horn.path_length,
            horn.n_segments,
            horn.hyperbolic_t,
        )
        bends = None
    elif horn.rectangular_segments:
        segments, bends, segment_widths = discretise_rectangular_segments(
            horn.rectangular_segments,
            n_per_segment=15,
        )
    elif horn.discretisation == "geometry" and horn.conical_segments:
        from pyhorn_core.solver.geometry_discretise import discretise_geometry_aware
        segments, bends_ext, bend_positions = discretise_geometry_aware(
            horn.conical_segments,
            horn.width,
            horn.bend_angles,
        )
        bends = None
    elif horn.sections:
        all_sections_segments = []
        for sec in horn.sections:
            n_seg = max(horn.n_segments, 10)
            segs = discretise_profile(
                sec.profile_type,
                sec.start_area,
                sec.end_area,
                sec.length,
                n_seg,
                hyperbolic_t=sec.hyperbolic_t if sec.hyperbolic_t is not None else 1.0,
            )
            all_sections_segments.extend(segs)
        segments = all_sections_segments
        bends = None
        mouth_area = horn.sections[-1].end_area
        if is_blh and horn.path_diff == 0.0:
            path_diff = sum(sec.length for sec in horn.sections)
    elif horn.conical_segments:
        segments, bends = discretise_conical_segments(
            horn.conical_segments, horn.width, n_per_segment=15
        )
    else:
        segments = horn.segments
        bends = horn.bends

    mouth_area = segments[-1][1] if segments else horn.mouth_area
    # Effective horn input area (throat) — used for area-step calculations.
    # For sections format the first segment's area reflects the user's start_area;
    # for legacy formats it is the first discretised segment area.
    horn_throat_area = (
        horn.throat_area
        if horn.throat_area > 0
        else (segments[0][1] if segments else horn.mouth_area)
    )

    segments = _merge_small_throat_segments(
        segments, bends, bends_ext, bend_positions, n_throat=20, min_len_merge=0.002
    )

    mouth_w = None
    mouth_h = None
    if horn.rectangular_segments:
        mouth_w = horn.rectangular_segments[-1][2]
        mouth_h = horn.rectangular_segments[-1][3]
    elif horn.width and horn.conical_segments:
        mouth_w = horn.width
        mouth_h = horn.conical_segments[-1][1]

    Zc_mouth = Z0 / mouth_area
    a_mouth = np.sqrt(mouth_area / np.pi)
    Zc_sd = Z0 / driver.sd
    a_sd = np.sqrt(driver.sd / np.pi)

    spl_out = np.zeros(len(freqs))
    z_in_out = np.zeros(len(freqs), dtype=complex)
    exc_out = np.zeros(len(freqs))
    total_pressure_out = np.zeros(len(freqs), dtype=complex)
    direct_pressure_out = np.zeros(len(freqs), dtype=complex) if (is_blh or is_tl) else None
    horn_pressure_out = np.zeros(len(freqs), dtype=complex) if (is_blh or is_tl) else None
    throat_impedance_out = np.zeros(len(freqs), dtype=complex)
    efficiency_out = np.zeros(len(freqs))
    electrical_input_power_out = np.zeros(len(freqs))
    cone_velocity_out = np.zeros(len(freqs))
    cone_acceleration_out = np.zeros(len(freqs))
    diaphragm_pressure_horn_side_out = np.zeros(len(freqs), dtype=complex) if (is_blh or is_tl) else None
    diaphragm_pressure_direct_side_out = np.zeros(len(freqs), dtype=complex) if (is_blh or is_tl) else None
    particle_velocity_throat_out = np.zeros(len(freqs))
    particle_velocity_mouth_out = np.zeros(len(freqs))
    particle_velocity_port_out = np.zeros(len(freqs))
    acoustic_power_out = np.zeros(len(freqs))
    spl_power_based_out = np.zeros(len(freqs))
    # Pre-compute frequency-dependent sensitivity_db (interpolates if needed)
    sensitivity_at_freqs = driver.get_sensitivity_db(freqs)

    for idx, f in enumerate(freqs):
        w = 2 * np.pi * f

        matrices = []

        # Throat chamber compliance: models the acoustic compliance of the volume
        # between the rear chamber and the horn throat (Vtc).  When Vtc <= 0 the
        # chamber is absent and this element is a passthrough (identity matrix).
        # NOTE: Hornresp uses AT=0.91 (throat-adapter constant) which changes how
        # the throat-chamber boundary is modeled — see GH-issues for the deeper
        # TMM formulation discrepancy at LF.
        if horn.vtc > 1e-14:
            matrices.append(
                compliance_tmatrix(f, horn.vtc, fr=horn.fr_tc, area=horn.atc)
            )

        if horn.ap1 > 0:
            matrices.append(
                throat_adapter_tmatrix(horn, f, fr=0.0)
            )
            # When lpt = 0, the throat adapter T-matrix is identity (zero-length tube).
            # Any area mismatch between ap1 and the horn's throat_area becomes a
            # phantom step — add it explicitly here.  For lpt > 0 the tube T-matrix
            # already carries the correct characteristic impedance; no extra step needed.
            if horn.lpt == 0 and abs(horn.ap1 - horn_throat_area) > 1e-12:
                matrices.append(
                    area_step_tmatrix(
                        f,
                        horn.ap1,
                        horn_throat_area,
                        lem_model=lem_step_model,
                        lem_strength=lem_step_strength,
                        lem_resistance=lem_step_resistance,
                    )
                )

        bend_pos_set = set(bend_positions) if bend_positions else None
        for i, seg in enumerate(segments):
            L = seg[0]
            S = seg[1]
            fr_seg = seg[2] if len(seg) >= 3 else 0.0

            matrices.append(tube_segment_tmatrix(f, L, S, fr_seg))

            if (
                bend_pos_set is not None
                and bends_ext is not None
                and bend_positions is not None
                and i in bend_pos_set
            ):
                b_idx = bend_positions.index(i)
                a1, a2, angle = bends_ext[b_idx]
                matrices.append(
                    area_step_tmatrix(
                        f,
                        a1,
                        a2,
                        lem_model=lem_step_model,
                        lem_strength=lem_step_strength,
                        lem_resistance=lem_step_resistance,
                    )
                )
                if angle > 0.01:
                    matrices.append(bend_tmatrix(f, (a1 + a2) / 2.0, angle))
            elif bends and i < len(bends):
                matrices.append(
                    area_step_tmatrix(
                        f,
                        bends[i][0],
                        bends[i][1],
                        lem_model=lem_step_model,
                        lem_strength=lem_step_strength,
                        lem_resistance=lem_step_resistance,
                    )
                )

        T = cascade(matrices)
        Zrad = radiation_impedance(
            f,
            mouth_area,
            horn.ang,
            _Zc=Zc_mouth,
            _a=a_mouth,
            mouth_width=mouth_w,
            mouth_height=mouth_h,
            mouth_radiation=horn.mouth_radiation,
        )

        Z_throat = (T[0, 0] * Zrad + T[0, 1]) / (T[1, 0] * Zrad + T[1, 1])
        throat_impedance_out[idx] = Z_throat

        Zrad_front = radiation_impedance(f, driver.sd, horn.ang, _Zc=Zc_sd, _a=a_sd)

        if horn.vented_box is not None and horn.vrc > 0:
            vb = horn.vented_box
            Z_vb = vented_box_impedance(f, vb.vrc, vb.lrc, vb.fr, ql=vb.ql)
            C_vb = horn.vrc / (RHO * C**2)
            Z_box = 1.0 / (1j * w * C_vb)
            if abs(Z_vb) < 1e-12 and abs(Z_box) < 1e-12:
                Z_ab = 0.0j
            elif abs(Z_vb) < 1e-12:
                Z_ab = Z_box
            elif abs(Z_box) < 1e-12:
                Z_ab = Z_vb
            else:
                Z_ab = (Z_vb * Z_box) / (Z_vb + Z_box)
        elif horn.passive_radiator is not None and horn.vrc > 0:
            pr = horn.passive_radiator
            Z_ab = passive_radiator_impedance(
                f, horn.vrc, pr.mma, pr.total_sp, ql_pr=pr.ql_pr
            )
        elif horn.slavbas is not None and horn.vrc > 0:
            sb = horn.slavbas
            # Compute rleak: either directly from sb.rleak, or derive from aleak + lrc
            if sb.rleak > 0:
                rleak = sb.rleak
            elif sb.aleak > 0 and sb.lrc > 0:
                # rleak = ρ·c·lrc / aleak²  (acoustic resistance of a short tube)
                rleak = RHO * C * sb.lrc / (sb.aleak ** 2)
            else:
                # No leak specified → fall back to standard sealed box
                rleak = 0.0
            Z_ab = slavbas_impedance(f, horn.vrc, rleak)
        else:
            # fr_tuning: explicit value from RearChamber if set (> 0),
            # otherwise derive from driver.fs (Hornresp BLH default).
            rc = getattr(horn, "rear_chamber", None)
            fr_tuning = getattr(rc, "fr_tuning", 0.0) if rc else 0.0
            if fr_tuning <= 0:
                fr_tuning = getattr(driver, "fs", 0.0)
            Z_ab = rear_chamber_impedance(
                f, horn.vrc, horn.lrc, fr=horn.fr_rc,
                chamber_type=getattr(rc, "chamber_type", "sealed") if rc else "sealed",
                fr_tuning=fr_tuning,
                throat_area=horn.throat_area,
            )

        if is_blh:
            Z_front_load = Zrad_front
            Z_rear_load = Z_throat + Z_ab
        elif is_tl:
            # TL: driver front loads into the TMM path (like non-BLH), rear loads into chamber.
            # The key difference from FLH is that direct radiation is NOT summed at 1m.
            Z_front_load = Z_throat
            Z_rear_load = Z_ab
        else:
            Z_front_load = Z_throat
            Z_rear_load = Z_ab

        Z_e, Z_ms, Z_me, Z_mt = _driver_impedance(
            driver.bl, driver.re, driver.le,
            driver.mms, driver.cms, driver.rms, driver.sd,
            f,
            le_freq_dependency=driver.le_freq_dependency,
            le_f_ref=driver.le_f_ref,
            lossy_le=driver.lossy_le,
            le_R_e_eddy=driver.le_R_e_eddy,
            le_f_lossy_ref=driver.le_f_lossy_ref,
        )

        Z_ad = Z_mt / (driver.sd**2)

        Z_tot = Z_ad + Z_front_load + Z_rear_load

        P_source = (driver.bl * driver.voltage) / (Z_e * driver.sd)

        U_driver = _velocity(P_source, Z_tot)

        v_driver = U_driver / driver.sd
        x_driver = _displacement(v_driver, f)
        exc_out[idx] = _excursion(x_driver)
        cone_velocity_out[idx] = np.abs(v_driver)
        cone_acceleration_out[idx] = np.abs(1j * w * v_driver)

        Z_mech_total_load = Z_ms + (Z_front_load + Z_rear_load) * (driver.sd**2)
        Z_in = Z_e + (driver.bl**2) / Z_mech_total_load
        z_in_out[idx] = Z_in

        if is_blh:
            U_throat = -U_driver
        elif is_tl:
            # TL: driver at closed end of line, cone faces into the line (same orientation as BLH).
            # Volume velocity into the line = -U_driver (cone pushes into line when moving forward).
            U_throat = -U_driver
        else:
            U_throat = U_driver

        P_throat = U_throat * Z_throat

        with np.errstate(divide='ignore', invalid='ignore'):
            denom = T[0, 0] + T[0, 1] / Zrad
            P_mouth = P_throat / denom if abs(denom) > 1e-12 else 0.0
            U_mouth = P_mouth / Zrad if abs(Zrad) > 1e-12 else 0.0

        throat_area = horn.throat_area
        mouth_area = horn.mouth_area
        particle_velocity_throat_out[idx] = np.abs(U_throat) / throat_area if throat_area > 0 else 0.0
        particle_velocity_mouth_out[idx] = np.abs(U_mouth) / mouth_area if mouth_area > 0 else 0.0

        U_port_local = 0.0j
        A_port_local = 1.0

        if is_blh:
            ang_direct = 2.0 * np.pi
            p_direct = 1j * w * RHO / ang_direct * U_driver * np.exp(-1j * w / C * 1.0)
            path_m = 1.0 + path_diff
            p_horn = 1j * w * RHO / horn.ang * U_mouth * np.exp(-1j * w / C * path_m)

            p_port_rad = 0.0j
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_horn_charged
                and horn.vrc > 0
            ):
                vb = horn.vented_box
                A_port_iter = (2 * np.pi * vb.fr / C) ** 2 * vb.vrc * vb.lrc
                A_port_iter = max(A_port_iter, 1e-6)
                for _ in range(5):
                    a_pipe_iter = np.sqrt(A_port_iter / np.pi)
                    L_eff_iter = vb.lrc + 0.6 * a_pipe_iter
                    A_port_iter = (2 * np.pi * vb.fr / C) ** 2 * vb.vrc * L_eff_iter
                    A_port_iter = max(A_port_iter, 1e-6)
                A_port = A_port_iter

                C_vb_box = vb.vrc / (RHO * C**2)
                Y_c = 1j * w * C_vb_box
                Y_vb_val = 1.0 / Z_vb if abs(Z_vb) > 1e-12 else 0.0j
                Y_total = Y_c + Y_vb_val
                P_box = U_driver / Y_total if abs(Y_total) > 1e-12 else 0.0j
                U_port = P_box / Z_vb if abs(Z_vb) > 1e-12 else 0.0j
                U_port_local = U_port
                A_port_local = A_port
                Z_rad_port = radiation_impedance(f, A_port, 2.0 * np.pi)
                p_port_rad = (
                    1j * w * RHO * U_port / (2.0 * np.pi) * np.exp(-1j * w / C * 1.0)
                )
                if vb.path_length_difference != 0.0:
                    p_port_rad *= np.exp(
                        -1j
                        * 2
                        * np.pi
                        * vb.path_length_difference
                        / C
                        * f
                    )

            p_1m = p_direct + p_horn + p_port_rad

            # ── Finite Transmission Line contribution ─────────────────────────
            p_tl_rad = 0.0j
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_transmission_line
                and horn.vented_box.ltl > 0
            ):
                vb = horn.vented_box
                # TL mouth radiation area: approximate as driver piston area
                A_tl = driver.sd
                Z_tl = transmission_line_impedance(f, vb.ltl, A_tl)
                if abs(Z_tl) > 1e-12:
                    # Parallel: TL + box compliance
                    C_vb_tl = vb.vrc / (RHO * C**2)
                    Y_c_tl = 1j * w * C_vb_tl
                    Y_tl = 1.0 / Z_tl
                    Y_total_tl = Y_c_tl + Y_tl
                    if abs(Y_total_tl) > 1e-12:
                        p_tl_input = U_driver / Y_total_tl
                        # Mouth velocity: U_mouth = 2·p_in·cos(k·l) / Z_rad_tl
                        k_tl = w / C
                        kL_tl = k_tl * vb.ltl
                        cos_kL_tl = np.cos(kL_tl)
                        Z_rad_tl = radiation_impedance(f, A_tl, 2.0 * np.pi)
                        if abs(Z_rad_tl) > 1e-12 and abs(cos_kL_tl) > 1e-12:
                            U_mouth_tl = 2.0 * p_tl_input * cos_kL_tl / Z_rad_tl
                            p_tl_rad = (
                                1j * w * RHO * U_mouth_tl
                                / (2.0 * np.pi)
                                * np.exp(-1j * w / C * 1.0)
                            )
                            # Phase shift for TL path length difference
                            p_tl_rad *= np.exp(-1j * w / C * vb.ltl)
                p_1m = p_direct + p_horn + p_port_rad + p_tl_rad
            # ── End Finite Transmission Line ──────────────────────────────────

            if direct_pressure_out is not None and horn_pressure_out is not None:
                direct_pressure_out[idx] = p_direct
                horn_pressure_out[idx] = p_horn

            p_throat = U_throat * Z_throat
            p_rear_chamber = U_driver * Z_ab
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_horn_charged
                and horn.vrc > 0
            ):
                p_rear_chamber = P_box
            if diaphragm_pressure_horn_side_out is not None:
                diaphragm_pressure_horn_side_out[idx] = p_throat
            if diaphragm_pressure_direct_side_out is not None:
                diaphragm_pressure_direct_side_out[idx] = p_rear_chamber
        elif is_tl:
            # Finite transmission line (TL): driver at closed end of horn/TL line.
            # The mouth is the ONLY acoustic output — direct radiation is excluded
            # from the summed SPL (unlike BLH).  The driver rear load (rear chamber
            # or sealed box) is handled via Z_ab, the same as non-BLH modes.
            p_direct = 0.0j  # TL: no direct radiation summing
            path_m = 1.0 + path_diff
            p_horn = 1j * w * RHO / horn.ang * U_mouth * np.exp(-1j * w / C * path_m)

            p_port_rad = 0.0j
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_horn_charged
                and horn.vrc > 0
            ):
                vb = horn.vented_box
                A_port_iter = (2 * np.pi * vb.fr / C) ** 2 * vb.vrc * vb.lrc
                A_port_iter = max(A_port_iter, 1e-6)
                for _ in range(5):
                    a_pipe_iter = np.sqrt(A_port_iter / np.pi)
                    L_eff_iter = vb.lrc + 0.6 * a_pipe_iter
                    A_port_iter = (2 * np.pi * vb.fr / C) ** 2 * vb.vrc * L_eff_iter
                    A_port_iter = max(A_port_iter, 1e-6)
                A_port = A_port_iter

                C_vb_box = vb.vrc / (RHO * C**2)
                Y_c = 1j * w * C_vb_box
                Y_vb_val = 1.0 / Z_vb if abs(Z_vb) > 1e-12 else 0.0j
                Y_total = Y_c + Y_vb_val
                P_box = U_driver / Y_total if abs(Y_total) > 1e-12 else 0.0j
                U_port = P_box / Z_vb if abs(Z_vb) > 1e-12 else 0.0j
                U_port_local = U_port
                A_port_local = A_port
                Z_rad_port = radiation_impedance(f, A_port, 2.0 * np.pi)
                p_port_rad = (
                    1j * w * RHO * U_port / (2.0 * np.pi) * np.exp(-1j * w / C * 1.0)
                )
                if vb.path_length_difference != 0.0:
                    p_port_rad *= np.exp(
                        -1j
                        * 2
                        * np.pi
                        * vb.path_length_difference
                        / C
                        * f
                    )

            p_1m = p_horn + p_port_rad  # mouth + port only, no direct

            # ── Finite Transmission Line contribution ─────────────────────────
            p_tl_rad = 0.0j
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_transmission_line
                and horn.vented_box.ltl > 0
            ):
                vb = horn.vented_box
                A_tl = driver.sd
                Z_tl = transmission_line_impedance(f, vb.ltl, A_tl)
                if abs(Z_tl) > 1e-12:
                    C_vb_tl = vb.vrc / (RHO * C**2)
                    Y_c_tl = 1j * w * C_vb_tl
                    Y_tl = 1.0 / Z_tl
                    Y_total_tl = Y_c_tl + Y_tl
                    if abs(Y_total_tl) > 1e-12:
                        p_tl_input = U_driver / Y_total_tl
                        k_tl = w / C
                        kL_tl = k_tl * vb.ltl
                        cos_kL_tl = np.cos(kL_tl)
                        Z_rad_tl = radiation_impedance(f, A_tl, 2.0 * np.pi)
                        if abs(Z_rad_tl) > 1e-12 and abs(cos_kL_tl) > 1e-12:
                            U_mouth_tl = 2.0 * p_tl_input * cos_kL_tl / Z_rad_tl
                            p_tl_rad = (
                                1j * w * RHO * U_mouth_tl
                                / (2.0 * np.pi)
                                * np.exp(-1j * w / C * 1.0)
                            )
                            p_tl_rad *= np.exp(-1j * w / C * vb.ltl)
                p_1m = p_horn + p_port_rad + p_tl_rad
            # ── End Finite Transmission Line ──────────────────────────────────

            if direct_pressure_out is not None and horn_pressure_out is not None:
                direct_pressure_out[idx] = p_direct
                horn_pressure_out[idx] = p_horn

            p_throat = U_throat * Z_throat
            p_rear_chamber = U_driver * Z_ab
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_horn_charged
                and horn.vrc > 0
            ):
                p_rear_chamber = P_box
            if diaphragm_pressure_horn_side_out is not None:
                diaphragm_pressure_horn_side_out[idx] = p_throat
            if diaphragm_pressure_direct_side_out is not None:
                diaphragm_pressure_direct_side_out[idx] = p_rear_chamber
        else:
            p_horn = 1j * w * RHO / horn.ang * U_mouth * np.exp(-1j * w / C * 1.0)

            p_port_rad = 0.0j
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_horn_charged
                and horn.vrc > 0
            ):
                vb = horn.vented_box
                Z_vb_val = vented_box_impedance(
                    f, vb.vrc, vb.lrc, vb.fr, ql=vb.ql
                )
                A_port_iter = (2 * np.pi * vb.fr / C) ** 2 * vb.vrc * vb.lrc
                A_port_iter = max(A_port_iter, 1e-6)
                for _ in range(5):
                    a_pipe_iter = np.sqrt(A_port_iter / np.pi)
                    L_eff_iter = vb.lrc + 0.6 * a_pipe_iter
                    A_port_iter = (2 * np.pi * vb.fr / C) ** 2 * vb.vrc * L_eff_iter
                    A_port_iter = max(A_port_iter, 1e-6)
                A_port = A_port_iter

                C_vb_box = vb.vrc / (RHO * C**2)
                Y_c = 1j * w * C_vb_box
                Y_vb_val = 1.0 / Z_vb_val if abs(Z_vb_val) > 1e-12 else 0.0j
                Y_total = Y_c + Y_vb_val
                P_box = U_driver / Y_total if abs(Y_total) > 1e-12 else 0.0j
                U_port = P_box / Z_vb_val if abs(Z_vb_val) > 1e-12 else 0.0j
                U_port_local = U_port
                A_port_local = A_port
                Z_rad_port = radiation_impedance(f, A_port, 2.0 * np.pi)
                p_port_rad = (
                    1j * w * RHO * U_port / (2.0 * np.pi) * np.exp(-1j * w / C * 1.0)
                )
                if vb.path_length_difference != 0.0:
                    p_port_rad *= np.exp(
                        -1j
                        * 2
                        * np.pi
                        * vb.path_length_difference
                        / C
                        * f
                    )

            p_1m = p_horn + p_port_rad

            # ── Finite Transmission Line contribution ─────────────────────────
            p_tl_rad = 0.0j
            if (
                horn.vented_box is not None
                and horn.vented_box.finite_transmission_line
                and horn.vented_box.ltl > 0
            ):
                vb = horn.vented_box
                A_tl = driver.sd
                Z_tl = transmission_line_impedance(f, vb.ltl, A_tl)
                if abs(Z_tl) > 1e-12:
                    C_vb_tl = vb.vrc / (RHO * C**2)
                    Y_c_tl = 1j * w * C_vb_tl
                    Y_tl = 1.0 / Z_tl
                    Y_total_tl = Y_c_tl + Y_tl
                    if abs(Y_total_tl) > 1e-12:
                        p_tl_input = U_driver / Y_total_tl
                        k_tl = w / C
                        kL_tl = k_tl * vb.ltl
                        cos_kL_tl = np.cos(kL_tl)
                        Z_rad_tl = radiation_impedance(f, A_tl, 2.0 * np.pi)
                        if abs(Z_rad_tl) > 1e-12 and abs(cos_kL_tl) > 1e-12:
                            U_mouth_tl = 2.0 * p_tl_input * cos_kL_tl / Z_rad_tl
                            p_tl_rad = (
                                1j * w * RHO * U_mouth_tl
                                / (2.0 * np.pi)
                                * np.exp(-1j * w / C * 1.0)
                            )
                            p_tl_rad *= np.exp(-1j * w / C * vb.ltl)
                p_1m = p_horn + p_port_rad + p_tl_rad
            # ── End Finite Transmission Line ──────────────────────────────────

        particle_velocity_port_out[idx] = np.abs(U_port_local) / A_port_local if A_port_local > 1e-12 else 0.0

        total_pressure_out[idx] = p_1m
        spl_val = _pressure_to_spl(np.array([p_1m]))[0]
        spl_out[idx] = spl_val

        z_in = z_in_out[idx]
        z_e_sq_mag = driver.re**2 + w * driver.le**2
        p_elec = driver.voltage**2 * z_in.real / z_e_sq_mag

        p_ac_mouth = np.abs(U_mouth) ** 2 * Zrad.real
        p_acoustic_direct = np.abs(U_driver) ** 2 * Zrad_front.real if is_blh else 0.0
        # Apply sensitivity_db calibration at the power level: only the horn-side
        # acoustic power is calibrated (it is the component modeled via the TMM).
        # The direct-radiation acoustic power is added uncalibrated.
        p_acoustic_horn_cal = p_ac_mouth * 10 ** (sensitivity_at_freqs[idx] / 10.0)
        p_acoustic = p_acoustic_horn_cal + p_acoustic_direct

        efficiency_out[idx] = (100.0 * p_acoustic / p_elec) if p_elec > 1e-12 else 0.0
        electrical_input_power_out[idx] = p_elec
        acoustic_power_out[idx] = p_ac_mouth  # mouth radiation power (matches docstring)
        # CRIT-3 fix: dB/W/m SPL using acoustic power.
        # P_REF = 10^-12 W gives 10*log10(P_acoustic/1e-12) = 10*log10(P_acoustic) + 120 dB.
        # sensitivity_db calibrates pyhorn's absolute HF level to Hornresp's dB/W/m reference.
        # Default sensitivity_db=0.0 gives the raw acoustic-power-based level.
        # For FE166NV2 at V=2.83 matching Hornresp: sensitivity_db ≈ -15.0 dB (HF band).
        # CALIBRATION AT POWER LEVEL: sensitivity_db is applied to P_horn first,
        # then P_direct is added uncalibrated: P_total = P_horn_cal + P_direct.
        if p_acoustic > 1e-15 and p_elec > 1e-12:
            spl_power_based_out[idx] = acoustic_power_to_spl_dB_W_m(
                p_acoustic, sensitivity_db=0.0
            )
        else:
            spl_power_based_out[idx] = 0.0

    # ── Measured driver SPL override ───────────────────────────────────────
    # Scale the direct-cone pressure to match the driver's measured SPL curve
    # (captures cone breakup that the lumped TS model cannot predict).
    # Phase is preserved from the model.  Total pressure and pressure-based
    # SPL are recomputed.  The "Total" displayed in the response plot is then
    # power-summed below from this corrected direct + the calibrated dB/W/m
    # horn-side level (see "Composite Total" block).
    if direct_pressure_out is not None:
        measured_db = driver.get_spl_response(freqs)
        if measured_db is not None:
            current_db = _pressure_to_spl(direct_pressure_out)
            with np.errstate(divide="ignore", invalid="ignore"):
                scale = np.power(10.0, (measured_db - current_db) / 20.0)
            direct_pressure_out = direct_pressure_out * scale
            if horn_pressure_out is not None:
                total_pressure_out = direct_pressure_out + horn_pressure_out
            else:
                total_pressure_out = direct_pressure_out
                
            new_spl_out = _pressure_to_spl(total_pressure_out)
            if spl_power_based_out is not None:
                spl_power_based_out += (new_spl_out - spl_out)
            
            spl_out = new_spl_out

    phase = np.unwrap(np.angle(total_pressure_out))
    omega_arr = 2 * np.pi * freqs
    if len(freqs) >= 2:
        group_delay_ms = -np.gradient(phase, omega_arr) * 1000.0
    else:
        group_delay_ms = np.zeros_like(freqs, dtype=float)
    group_delay_per_period_out = group_delay_ms / 1000.0 * freqs
    numerical_artifacts = _detect_numerical_artifacts(freqs, spl_out)

    all_artifacts = list(numerical_artifacts)
    if len(freqs) >= 11:
        window = 5
        for i in range(window, len(spl_out) - window):
            local_median = float(np.median(np.concatenate([spl_out[i - window:i], spl_out[i + 1:i + window + 1]])))
            if spl_out[i] - local_median > 10.0 and spl_out[i] - spl_out[i - 1] > 3.0 and spl_out[i] - spl_out[i + 1] > 3.0:
                all_artifacts.append(float(freqs[i]))

    if all_artifacts:
        spl_out = _smooth_spl_near_artifacts(freqs, spl_out, all_artifacts)

    spl_notched_out: Optional[np.ndarray] = None
    if notch_filter and notch_frequencies:
        spl_notched_out = _apply_notch_filter(freqs, spl_out, notch_frequencies, notch_q)

    _DEFAULT_OFF_AXIS_ANGLES = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0])
    _off_axis_angles = np.asarray(off_axis_angles) if off_axis_angles is not None else _DEFAULT_OFF_AXIS_ANGLES
    n_angles = len(_off_axis_angles)

    fdd_di_out: Optional[np.ndarray] = None
    direction_index_out: Optional[np.ndarray] = None

    if fdd_mode:
        fdd_di_out = _fdd_directivity_index(freqs, mouth_area, f_c=fdd_fc, D_max=fdd_dmax)
        off_axis_spl_out = _fdd_off_axis_spl(
            freqs, mouth_area, _off_axis_angles, f_c=fdd_fc, D_max=fdd_dmax
        )
        radiation_angle = _fdd_radiation_angle(freqs, mouth_area, off_axis_spl_out, _off_axis_angles, f_c=fdd_fc)
        a_mouth_piston = np.sqrt(mouth_area / np.pi)
        k_arr = 2.0 * np.pi * freqs / C
        ka_arr = k_arr * a_mouth_piston
        direction_index_out = np.zeros((len(freqs), n_angles), dtype=float)
        for j, ang_deg in enumerate(_off_axis_angles):
            ang_rad = np.radians(ang_deg)
            sin_t = np.sin(ang_rad)
            x = ka_arr * sin_t
            x_safe = np.where(x < 0.05, 0.05, x)
            j1_vals = jv(1, x_safe)
            sinc_unsq = 2.0 * j1_vals / (x_safe + 1e-12)
            direction_factor = np.where(ka_arr < 0.05, 1.0, sinc_unsq)
            direction_index_out[:, j] = 10.0 * np.log10(np.maximum(direction_factor, 1e-12))
    else:
        off_axis_spl_out = np.zeros((len(freqs), n_angles))
        a_mouth_piston = np.sqrt(mouth_area / np.pi)
        k_arr = 2.0 * np.pi * freqs / C
        ka_arr = k_arr * a_mouth_piston

        for j, ang_deg in enumerate(_off_axis_angles):
            ang_rad = np.radians(ang_deg)
            sin_t = np.sin(ang_rad)
            x = ka_arr * sin_t
            x_safe = np.where(x < 0.05, 0.05, x)
            j1_vals = jv(1, x_safe)
            sinc_sq = (2.0 * j1_vals / (x_safe + 1e-12)) ** 2
            D = np.where(ka_arr < 0.05, 1.0, sinc_sq)
            off_axis_spl_out[:, j] = 10.0 * np.log10(D + 1e-12)

        direction_index_out = np.zeros((len(freqs), n_angles), dtype=float)
        for j, ang_deg in enumerate(_off_axis_angles):
            ang_rad = np.radians(ang_deg)
            sin_t = np.sin(ang_rad)
            x = ka_arr * sin_t
            x_safe = np.where(x < 0.05, 0.05, x)
            j1_vals = jv(1, x_safe)
            sinc_unsq = 2.0 * j1_vals / (x_safe + 1e-12)
            direction_factor = np.where(ka_arr < 0.05, 1.0, sinc_unsq)
            direction_index_out[:, j] = 10.0 * np.log10(np.maximum(direction_factor, 1e-12))

        radiation_angle: Optional[float] = None
        if n_angles >= 2:
            angles_6db = []
            for i in range(len(freqs)):
                rel_spl_at_angle = off_axis_spl_out[i, :]
                below_6db = np.where(rel_spl_at_angle <= -6.0)[0]
                if len(below_6db) > 0:
                    idx_below = below_6db[0]
                    if idx_below == 0:
                        ang_6db = float(_off_axis_angles[0])
                    else:
                        idx_above = idx_below - 1
                        s1 = float(_off_axis_angles[idx_above])
                        s2 = float(_off_axis_angles[idx_below])
                        v1 = rel_spl_at_angle[idx_above]
                        v2 = rel_spl_at_angle[idx_below]
                        if abs(v2 - v1) > 1e-9:
                            ang_6db = s1 + (s2 - s1) * (-6.0 - v1) / (v2 - v1)
                        else:
                            ang_6db = s1
                    angles_6db.append(ang_6db)
            if angles_6db:
                valid = [a for a in angles_6db if a < 88.0]
                if valid:
                    radiation_angle = float(np.mean(valid))

    is_finite_horn_charged = (
        horn.vented_box is not None and horn.vented_box.finite_horn_charged and horn.vrc > 0
    )

    second_tone_distortion = (
        _compute_second_tone_distortion(freqs, driver, horn, spl_out)
        if compute_distortion and _is_single_segment_horn(horn)
        else None
    )

    thermal_compression_db: Optional[np.ndarray] = None
    if _thermal_T_voice is not None and _thermal_T_voice > 20.0:
        thermal_compression_db = compute_thermal_power_compression(
            freqs, driver, horn, T_voice=_thermal_T_voice
        )

    # Futtrup audible group delay limit (Hornresp page 113):
    # GDlimit = 1000 × 1160.6 / (5643 × f^0.81511 − f)  [ms]
    # Below ~50 Hz the denominator approaches zero; clamp to a safe upper bound.
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = 5643.0 * freqs**0.81511 - freqs
        futtrup_gdlimit = np.where(
            denominator > 1.0,
            1000.0 * 1160.6 / denominator,
            1000.0 * 1160.6 / 1.0,
        )

    return SimulationResult(
        freqs=freqs,
        spl=spl_out,
        impedance=z_in_out,
        excursion=exc_out,
        segments=segments,
        ib_spl=infinite_baffle_response(freqs, driver),
        direct_spl=(
            _pressure_to_spl(direct_pressure_out)
            if direct_pressure_out is not None
            else None
        ),
        horn_spl=(
            _pressure_to_spl(horn_pressure_out)
            if horn_pressure_out is not None
            else None
        ),
        group_delay=group_delay_ms,
        group_delay_per_period=group_delay_per_period_out,
        phase=phase,
        pressure=total_pressure_out,
        throat_impedance=throat_impedance_out,
        impedance_phase_deg=np.angle(z_in_out) * 180.0 / np.pi,
        segment_widths=segment_widths,
        numerical_artifacts=numerical_artifacts,
        efficiency_pct=efficiency_out,
        electrical_input_power=electrical_input_power_out,
        off_axis_spl=off_axis_spl_out,
        off_axis_angles=_off_axis_angles,
        radiation_angle=radiation_angle,
        finite_horn_charged=is_finite_horn_charged,
        second_tone_distortion=second_tone_distortion,
        thermal_compression_db=thermal_compression_db,
        spl_notched=spl_notched_out,
        fdd_enabled=fdd_mode,
        fdd_di=fdd_di_out,
        direction_index=direction_index_out,
        cone_velocity=cone_velocity_out,
        cone_acceleration=cone_acceleration_out,
        diaphragm_pressure_total=(
            diaphragm_pressure_horn_side_out - diaphragm_pressure_direct_side_out
            if diaphragm_pressure_horn_side_out is not None
            else None
        ),
        diaphragm_pressure_horn_side=diaphragm_pressure_horn_side_out,
        diaphragm_pressure_direct_side=diaphragm_pressure_direct_side_out,
        particle_velocity_throat=particle_velocity_throat_out,
        particle_velocity_mouth=particle_velocity_mouth_out,
        particle_velocity_port=particle_velocity_port_out,
        futtrup_gdlimit=futtrup_gdlimit,
        acoustic_power=acoustic_power_out,
        spl_power_based=spl_power_based_out,
    )
