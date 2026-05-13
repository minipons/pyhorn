"""Resize Wizard — proportional horn + driver scaling (Hornresp page 68).

Scale horn geometry and driver Sd proportionally.
Larger horn → response shifts to lower frequencies; curve shape is preserved.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Dict, Any

from pyhorn_core.config.models import DriverSpecs, HornGeometry


@dataclass
class ResizeWizard:
    """Options for the Resize Wizard.

    resize_factor: linear scale factor (e.g. 1.5 = 50% larger in every linear dim).
                   > 1  → larger horn, lower cutoff frequency.
                   < 1  → smaller horn, higher cutoff frequency.
    adjust_sd:     scale driver piston area (Sd) by resize_factor².
                   Disable this to keep the original driver Sd (e.g. same driver,
                   different horn size).
    adjust_re:     scale driver voice coil resistance (Re) by resize_factor².
                   Physically, Re of the same driver does NOT change with size —
                   leave False (the default). Set True if swapping to a different
                   driver sized to match the scaled horn.
    """
    resize_factor: float
    adjust_sd: bool = True
    adjust_re: bool = False  # Re doesn't change with size for the same driver


def _scale_horn_geometry(g: HornGeometry, factor: float) -> HornGeometry:
    """Apply proportional scaling to a HornGeometry object in-place (via replace)."""
    factor2 = factor * factor
    factor3 = factor2 * factor

    # ── Simple scalar fields ───────────────────────────────────────────────
    updates: Dict[str, Any] = {}

    if g.throat_area > 0:
        updates["throat_area"] = g.throat_area * factor2
    if g.mouth_area > 0:
        updates["mouth_area"] = g.mouth_area * factor2
    if g.path_length > 0:
        updates["path_length"] = g.path_length * factor
    if g.lpt > 0:
        updates["lpt"] = g.lpt * factor
    if g.ap1 > 0:
        updates["ap1"] = g.ap1 * factor2
    if g.vrc > 0:
        updates["vrc"] = g.vrc * factor3
    if g.vtc > 0:
        updates["vtc"] = g.vtc * factor3
    if g.lrc > 0:
        updates["lrc"] = g.lrc * factor
    if g.atc > 0:
        updates["atc"] = g.atc * factor2

    # ── conical_segments: (dim_start, dim_end, length_m, [fr])
    #     linear dims × factor (area dims when width=None → area scales by factor²)
    #     length × factor
    if g.conical_segments:
        scaled = []
        for seg in g.conical_segments:
            if len(seg) >= 3:
                d_start = seg[0] * factor
                d_end = seg[1] * factor
                length = seg[2] * factor
                rest = tuple(seg[3:])  # flow resistivity unchanged
                scaled.append((d_start, d_end, length) + rest)
            else:
                scaled.append(seg)
        updates["conical_segments"] = scaled

    # ── rectangular_segments: (w_start, h_start, w_end, h_end, length_m, [fr])
    #     width stays constant (cabinet width is a design constraint)
    #     height × factor² → area changes by factor² (w × h_new = w × h × factor²)
    #     length × factor
    if g.rectangular_segments:
        scaled = []
        for seg in g.rectangular_segments:
            if len(seg) >= 5:
                w_start = seg[0]            # unchanged
                h_start = seg[1] * factor2  # area dim → ×factor²
                w_end = seg[2]              # unchanged
                h_end = seg[3] * factor2    # area dim → ×factor²
                length = seg[4] * factor
                rest = tuple(seg[5:])        # flow resistivity unchanged
                scaled.append((w_start, h_start, w_end, h_end, length) + rest)
            else:
                scaled.append(seg)
        updates["rectangular_segments"] = scaled

    # ── legacy segments: (length_m, area_m2, [fr])
    #     length × factor; area × factor²
    if g.segments:
        scaled = []
        for seg in g.segments:
            length = seg[0] * factor if len(seg) >= 1 else 0.0
            area = seg[1] * factor2 if len(seg) >= 2 else 0.0
            rest = tuple(seg[2:])
            scaled.append((length, area) + rest)
        updates["segments"] = scaled

    # ── bends: (area_before_m2, area_after_m2) → both × factor²
    if g.bends:
        updates["bends"] = [
            (a * factor2, b * factor2) for a, b in g.bends
        ]

    # ── coordinates: (x, y) → both × factor
    if g.coordinates:
        updates["coordinates"] = [(x * factor, y * factor) for x, y in g.coordinates]

    # ── enclosure_dims: (depth, height) → both × factor
    if g.enclosure_dims:
        updates["enclosure_dims"] = (
            g.enclosure_dims[0] * factor,
            g.enclosure_dims[1] * factor,
        )

    # ── driver_coord: (x, y) → both × factor
    if g.driver_coord:
        updates["driver_coord"] = (
            g.driver_coord[0] * factor,
            g.driver_coord[1] * factor,
        )

    # ── width: intentionally NOT scaled (design constraint: cabinet width fixed)
    # ── ang, profile_type, hyperbolic_t, n_segments, discretisation, bend_angles,
    #    enclosure_type, path_diff, fr_rc, fr_tc, throat_adapter_type,
    #    lem_step_model, lem_step_strength, lem_step_resistance,
    #    vented_box, passive_radiator: unchanged

    return replace(g, **updates) if updates else g


def _scale_driver(driver: DriverSpecs, factor: float, adjust_sd: bool, adjust_re: bool) -> DriverSpecs:
    """Scale driver specs proportionally."""
    factor2 = factor * factor

    updates = {}
    if adjust_sd:
        updates["sd"] = driver.sd * factor2
    # Note: Re does NOT physically change with size for the same driver.
    # Set adjust_re=True only when swapping to a driver sized for the scaled horn.
    if adjust_re:
        updates["re"] = driver.re * factor2
    # mms, bl, cms, rms, vas, fs, qts, qes, qms — unchanged (same driver Thiele-Small)
    # voltage, le, xmax, alpha_re, le_freq_dependency, le_f_ref — unchanged

    return replace(driver, **updates) if updates else driver


def apply_resize(
    geometry: HornGeometry,
    driver: DriverSpecs,
    resize_factor: float,
    adjust_sd: bool = True,
    adjust_re: bool = False,
) -> Tuple[HornGeometry, DriverSpecs]:
    """Apply proportional scaling to horn geometry and driver specs.

    Scaling rules (Hornresp page 68 — "the response of the speaker system can be
    shifted up or down in frequency without changing the curve shape by
    increasing or decreasing the size of the system"):

    ── Geometry ────────────────────────────────────────────────────────────────
    • throat_area  (S1)  : × resize_factor²
    • mouth_area   (S2)  : × resize_factor²
    • path_length  (L12) : × resize_factor
    • conical_segments   : linear dims × resize_factor; length × resize_factor
    • rectangular_segments: heights × resize_factor² (area dims), widths unchanged,
                            length × resize_factor
    • legacy segments     : length × resize_factor, area × resize_factor²
    • bends              : both area values × resize_factor²
    • lpt                : × resize_factor
    • ap1                : × resize_factor²
    • vrc, vtc           : × resize_factor³
    • atc                : × resize_factor²
    • lrc                : × resize_factor
    • coordinates        : (x, y) × resize_factor
    • enclosure_dims     : (depth, height) × resize_factor
    • driver_coord        : (x, y) × resize_factor
    • width               : UNCHANGED (cabinet width is a design constraint)

    ── Driver ───────────────────────────────────────────────────────────────────
    • sd          : × resize_factor²  (piston area) — disabled with --no-adjust-sd
    • re          : × resize_factor²  — disabled by default (set --adjust-re to enable)
    • mms, bl, cms, rms, vas, fs, qts, qes, qms : UNCHANGED
      (same driver; these are material/motor properties, not geometric sizes)
    • voltage, le, xmax, alpha_re : UNCHANGED

    Returns (resized_geometry, resized_driver).
    """
    factor = float(resize_factor)
    if factor <= 0:
        raise ValueError(f"resize_factor must be positive, got {factor}")

    resized_geometry = _scale_horn_geometry(geometry, factor)
    resized_driver = _scale_driver(driver, factor, adjust_sd, adjust_re)

    return resized_geometry, resized_driver
