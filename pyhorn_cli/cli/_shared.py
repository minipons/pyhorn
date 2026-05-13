"""Shared helpers and constants for pyhorn_cli commands."""

import math
from typing import Optional

import numpy as np
from scipy.special import j1

from pyhorn_core.solver.models import RHO, C
# pyhorn_fold is a separate package — stub out if not installed
try:
    from pyhorn_fold import (
        throat_chamber_side_length as _throat_chamber_side_length,
    )
except ImportError:
    def _throat_chamber_side_length(*args, **kwargs):
        return None


# Default off-axis angles for piston directivity computation
DEFAULT_OFF_AXIS_ANGLES = [0, 15, 30, 45, 60, 75, 90]

_PRESET_SUMMARY_FREQS = [250, 500, 1000, 2000, 4000]


def _piston_directivity_db(angle_deg: float, ka: float) -> float:
    """Levine/Inglis piston directivity: SPL reduction at angle θ vs on-axis.

    D(θ) = [2·J1(ka·sinθ) / (ka·sinθ)]²
    rel_SPL(θ) = 10·log10(D(θ)) dB relative to on-axis
    """
    if ka < 0.05:
        return 0.0
    angle_rad = math.radians(angle_deg)
    sin_t = math.sin(angle_rad)
    x = ka * sin_t
    if x < 0.05:
        return 0.0
    j1_val = j1(x)
    denom = x
    sinc = 2.0 * j1_val / denom
    D = sinc ** 2
    return 10.0 * math.log10(D + 1e-12)


def _compute_piston_off_axis_spl(freqs, mouth_area, angles):
    """Compute off-axis SPL (dB relative to on-axis) for each angle.

    Uses Levine/Inglis piston-in-baffle model at the horn mouth.
    """
    a = math.sqrt(mouth_area / math.pi)
    k = 2.0 * math.pi * freqs / C
    ka = k * a  # shape (n_freq,)
    results = {}
    for angle in angles:
        angle_rad = math.radians(angle)
        sin_t = math.sin(angle_rad)
        x = ka * sin_t
        x_safe = np.where(x < 0.05, 0.05, x)
        j1_vals = j1(x_safe)
        sinc_sq = (2.0 * j1_vals / (x_safe + 1e-12)) ** 2
        D = np.where(ka < 0.05, 1.0, sinc_sq)
        rel_spl = 10.0 * np.log10(D + 1e-12)
        results[str(angle)] = rel_spl
    return results


def _print_radiation_summary(freqs: np.ndarray, spl: np.ndarray, mouth_area: float):
    """Print a radiation summary table using the piston model."""
    import typer

    if mouth_area <= 0 or len(freqs) == 0:
        return
    a = math.sqrt(mouth_area / math.pi)
    k = 2.0 * math.pi * freqs / C
    ka = k * a

    typer.echo("\nRadiation Summary (piston model approximation)")
    typer.echo(f"{'Frequency':<12} {'On-axis SPL':<14} {'-6dB Beamwidth':<18} {'Directivity Index'}")
    typer.echo("-" * 60)

    for f_preset in _PRESET_SUMMARY_FREQS:
        idx = np.argmin(np.abs(freqs - f_preset))
        f_actual = freqs[idx]
        on_axis_spl_db = spl[idx]
        ka_val = float(ka[idx])

        # DI via numerical integration over 0-90°
        theta_arr = np.linspace(0, np.pi / 2, 500)
        sin_theta = np.sin(theta_arr)
        x_arr = ka_val * sin_theta
        x_safe = np.where(x_arr < 0.05, 0.05, x_arr)
        j1_arr = j1(x_safe)
        sinc_arr = 2.0 * j1_arr / (x_safe + 1e-12)
        D_arr = np.where(ka_val < 0.05, 1.0, sinc_arr ** 2)
        D_avg = np.trapezoid(D_arr * sin_theta, theta_arr)
        DI = 10.0 * math.log10(1.0 / D_avg) if D_avg > 0 else 0.0

        # -6dB beamwidth
        target_D = 0.251
        beamwidth = ">180°"
        if ka_val >= 0.05:
            bw_angles = np.linspace(0, 90, 500)
            x_bw = ka_val * np.sin(np.radians(bw_angles))
            x_bw_safe = np.where(x_bw < 0.05, 0.05, x_bw)
            j1_bw = j1(x_bw_safe)
            D_bw = (2.0 * j1_bw / (x_bw_safe + 1e-12)) ** 2
            D_bw = np.where(ka_val < 0.05, 1.0, D_bw)
            below = D_bw >= target_D
            if np.any(~below):
                first_null = bw_angles[~below][0]
                beamwidth = f"{first_null:.0f}°"

        freq_label = f"{f_preset} Hz"
        typer.echo(
            f"{freq_label:<12} {on_axis_spl_db:<14.1f} {beamwidth:<18} {DI:.1f} dB"
        )


def _folded_throat_chamber_side(horn) -> Optional[float]:
    if horn.width is None or horn.width <= 0 or horn.vtc <= 0:
        return None
    return _throat_chamber_side_length(horn, horn.width)


def _horn_geometry_to_dict(horn) -> dict:
    """Serialize a HornGeometry to a plain dict for YAML output."""
    def _round_list(lst, decimals=4):
        return [round(v, decimals) for v in lst] if lst else []

    data: dict = {
        "enclosure_type": horn.enclosure_type,
    }
    if horn.throat_area > 0:
        data["throat_area"] = round(horn.throat_area, 8)
    if horn.mouth_area > 0:
        data["mouth_area"] = round(horn.mouth_area, 8)
    if horn.path_length > 0:
        data["path_length"] = round(horn.path_length, 6)
    if horn.width is not None:
        data["width"] = round(horn.width, 6)
    if horn.lrc > 0:
        data["lrc"] = round(horn.lrc, 6)
    if horn.vrc > 0:
        data["vrc"] = round(horn.vrc, 8)
    if horn.vtc > 0:
        data["vtc"] = round(horn.vtc, 8)
    if horn.atc > 0:
        data["atc"] = round(horn.atc, 8)
    if horn.lpt > 0:
        data["lpt"] = round(horn.lpt, 6)
    if horn.ap1 > 0:
        data["ap1"] = round(horn.ap1, 8)
    if horn.throat_adapter_type and horn.throat_adapter_type != "cylindrical":
        data["throat_adapter_type"] = horn.throat_adapter_type
    if horn.enclosure_dims:
        data["enclosure_dims"] = _round_list(horn.enclosure_dims)
    if horn.driver_coord:
        data["driver_coord"] = _round_list(horn.driver_coord)
    if horn.coordinates:
        data["coordinates"] = [[round(x, 5), round(y, 5)] for x, y in horn.coordinates]
    if horn.conical_segments:
        data["conical_segments"] = [
            [round(v, 6) if i < 3 else v for i, v in enumerate(seg)]
            for seg in horn.conical_segments
        ]
    if horn.rectangular_segments:
        data["rectangular_segments"] = [
            [round(v, 6) if i < 5 else v for i, v in enumerate(seg)]
            for seg in horn.rectangular_segments
        ]
    if horn.segments:
        data["segments"] = [
            [round(v, 6) if i < 2 else v for i, v in enumerate(seg)]
            for seg in horn.segments
        ]
    if horn.bends:
        data["bends"] = [[round(a, 8), round(b, 8)] for a, b in horn.bends]
    if horn.discretisation:
        data["discretisation"] = horn.discretisation
    if horn.bend_angles:
        data["bend_angles"] = [round(a, 4) for a in horn.bend_angles]
    if horn.lem_step_model and horn.lem_step_model != "ideal":
        data["lem_step_model"] = horn.lem_step_model
        data["lem_step_strength"] = round(horn.lem_step_strength, 4)
    return data


def _driver_specs_to_dict(driver) -> dict:
    """Serialize a DriverSpecs to a plain dict for YAML output."""
    fields = [
        "fs", "qts", "qes", "qms", "vas", "re", "bl", "mms", "cms",
        "rms", "sd", "voltage", "le", "xmax", "alpha_re",
    ]
    data = {}
    for f in fields:
        val = getattr(driver, f, None)
        if val is not None and val != 0.0:
            data[f] = round(val, 8) if f != "alpha_re" else val
    if driver.le_freq_dependency:
        data["le_freq_dependency"] = True
        data["le_f_ref"] = round(driver.le_f_ref, 2)
    return data
