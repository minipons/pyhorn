"""
pyhorn_physics.bem_validation — BEM vs TMM validation framework.

Purpose
=======
This module provides tools to validate pyhorn's Transfer Matrix Method (TMM)
predictions against Boundary Element Method (BEM) reference data from external
tools (e.g. BEMPPSolver, horn-simulation with bempp-cl).

BEM is a numerically exact method for solving the Helmholtz equation on
arbitrary 3D geometries. It captures:
  - Full 3D wave interactions (TMM assumes plane waves within each segment)
  - Diffraction and edge effects at the mouth
  - Mutual coupling between throat and enclosure
  - Frequency-dependent directivity patterns

The validation framework enables:
  1. Loading BEM-computed reference data (SPL, impedance, directivity)
  2. Running pyhorn TMM predictions for the same geometry
  3. Computing delta (TMM − BEM) across frequency
  4. Statistical metrics: RMS error, max |delta|, mean delta
  5. Visual comparison plots

Usage
=====
::

    from pyhorn_physics.bem_validation import (
        load_bem_reference,
        compute_tmm_prediction,
        compare_spl,
        BemValidationResult,
    )

    # Load BEM reference data
    bem = load_bem_reference("tests/benchmarks/bem/exponential_horn/reference/spl.csv")

    # Compute TMM prediction
    tmm = compute_tmm_prediction(freqs, driver, horn)

    # Compare
    result = compare_spl(bem, tmm)
    print(f"RMS delta: {result.rmse_db:.2f} dB")
    print(f"Max |delta|: {result.max_abs_delta_db:.2f} dB")

Reference
=========
jw_report.md: "Add BEM validation mode — compare TMM predictions against BEM"
horn-simulation-report.md: "BEM-based exterior radiation model" recommendation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Literal

import numpy as np
from scipy.interpolate import interp1d


@dataclass
class BemReferenceData:
    """Container for BEM reference data from external simulation."""

    freqs: np.ndarray
    spl: Optional[np.ndarray] = None
    impedance_real: Optional[np.ndarray] = None
    impedance_imag: Optional[np.ndarray] = None
    directivity_horizontal: Optional[np.ndarray] = None
    directivity_vertical: Optional[np.ndarray] = None
    directivity_angles: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

    @property
    def impedance(self) -> Optional[np.ndarray]:
        """Complex impedance if real/imag parts available."""
        if self.impedance_real is not None and self.impedance_imag is not None:
            return self.impedance_real + 1j * self.impedance_imag
        return None

    def has_spl(self) -> bool:
        return self.spl is not None and len(self.spl) > 0

    def has_impedance(self) -> bool:
        return self.impedance is not None and len(self.impedance) > 0

    def has_directivity(self) -> bool:
        return (
            self.directivity_horizontal is not None
            or self.directivity_vertical is not None
        ) and self.directivity_angles is not None


@dataclass
class TmmPredictionData:
    """Container for TMM prediction data from pyhorn."""

    freqs: np.ndarray
    spl: np.ndarray
    impedance: Optional[np.ndarray] = None
    directivity_horizontal: Optional[np.ndarray] = None
    directivity_vertical: Optional[np.ndarray] = None
    directivity_angles: Optional[np.ndarray] = None


@dataclass
class BemValidationResult:
    """Results of comparing TMM prediction against BEM reference."""

    freqs: np.ndarray
    delta_spl: np.ndarray
    mean_delta_db: float
    std_delta_db: float
    rmse_db: float
    max_abs_delta_db: float
    max_delta_db: float
    min_delta_db: float

    delta_impedance_real: Optional[np.ndarray] = None
    delta_impedance_imag: Optional[np.ndarray] = None
    impedance_rmse: Optional[float] = None

    delta_directivity_horizontal: Optional[np.ndarray] = None
    delta_directivity_vertical: Optional[np.ndarray] = None
    directivity_rmse: Optional[float] = None

    pass_threshold: bool = True
    threshold_db: float = 3.0


def load_bem_reference(
    path: Path | str,
    format: Literal["csv", "json"] = "csv",
) -> BemReferenceData:
    """
    Load BEM reference data from a file.

    Supports CSV format with columns:
      - Freq (Hz)
      - SPL (dB)
      - Z_real (ohms, optional)
      - Z_imag (ohms, optional)
      - DI_horiz_0, DI_horiz_15, ... (directivity index at angles, optional)
      - DI_vert_0, DI_vert_15, ... (vertical directivity, optional)

    Parameters
    ----------
    path : Path or str
        Path to the reference data file.
    format : str
        File format: "csv" or "json".

    Returns
    -------
    BemReferenceData
        Parsed reference data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the format is not supported or data is malformed.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BEM reference file not found: {p}")

    if format == "csv":
        return _load_bem_csv(p)
    elif format == "json":
        return _load_bem_json(p)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'json'.")


def _load_bem_csv(p: Path) -> BemReferenceData:
    """Load BEM reference from CSV file, skipping comment lines."""
    freqs = []
    spl = []
    z_real = []
    z_imag = []
    directivity_horizontal = {}
    directivity_vertical = {}

    with open(p) as f:
        lines = [line for line in f if not line.strip().startswith("#")]
        reader = csv.DictReader(lines)
        for row in reader:
            freq_val = row.get("Freq (hertz)") or row.get("Freq") or row.get("freq") or "0"
            freqs.append(float(freq_val))
            s = row.get("SPL (dB)") or row.get("SPL") or row.get("spl") or ""
            spl.append(float(s) if s else np.nan)
            zr = row.get("Z_real") or row.get("Zreal") or row.get("z_real") or ""
            zi = row.get("Z_imag") or row.get("Zimag") or row.get("z_imag") or ""
            z_real.append(float(zr) if zr else np.nan)
            z_imag.append(float(zi) if zi else np.nan)

            for col, val in row.items():
                if col is None:
                    continue
                if col.startswith("DI_horiz_"):
                    try:
                        angle = int(col.split("_")[-1])
                        directivity_horizontal[angle] = float(val)
                    except (ValueError, IndexError):
                        pass
                elif col.startswith("DI_vert_"):
                    try:
                        angle = int(col.split("_")[-1])
                        directivity_vertical[angle] = float(val)
                    except (ValueError, IndexError):
                        pass

    freqs = np.array(freqs)
    spl = np.array(spl) if spl else None
    z_real = np.array(z_real) if z_real else None
    z_imag = np.array(z_imag) if z_imag else None

    di_horiz = None
    di_vert = None
    di_angles = None

    if directivity_horizontal:
        di_angles = np.array(sorted(directivity_horizontal.keys()))
        di_horiz = np.array([directivity_horizontal[a] for a in di_angles])
    if directivity_vertical:
        if di_angles is None:
            di_angles = np.array(sorted(directivity_vertical.keys()))
        di_vert = np.array([directivity_vertical[a] for a in di_angles])

    return BemReferenceData(
        freqs=freqs,
        spl=spl,
        impedance_real=z_real,
        impedance_imag=z_imag,
        directivity_horizontal=di_horiz,
        directivity_vertical=di_vert,
        directivity_angles=di_angles,
        metadata={"source": str(p), "format": "csv"},
    )


def _load_bem_json(p: Path) -> BemReferenceData:
    """Load BEM reference from JSON file."""
    with open(p) as f:
        data = json.load(f)

    freqs = np.array(data.get("freqs", []))
    spl = np.array(data.get("spl")) if "spl" in data else None
    z_real = np.array(data.get("impedance_real")) if "impedance_real" in data else None
    z_imag = np.array(data.get("impedance_imag")) if "impedance_imag" in data else None

    di_horiz = None
    di_vert = None
    di_angles = None

    if "directivity_horizontal" in data:
        di_horiz = np.array(data["directivity_horizontal"])
        di_angles = np.array(data.get("directivity_angles", []))
    if "directivity_vertical" in data:
        di_vert = np.array(data["directivity_vertical"])

    return BemReferenceData(
        freqs=freqs,
        spl=spl,
        impedance_real=z_real,
        impedance_imag=z_imag,
        directivity_horizontal=di_horiz,
        directivity_vertical=di_vert,
        directivity_angles=di_angles,
        metadata={"source": str(p), "format": "json"},
    )


def interpolate_to_reference(
    target_freqs: np.ndarray,
    source_freqs: np.ndarray,
    source_values: np.ndarray,
    kind: str = "linear",
) -> np.ndarray:
    """
    Interpolate source data to target frequency grid using log-frequency interpolation.

    Parameters
    ----------
    target_freqs : np.ndarray
        Desired frequency points (Hz).
    source_freqs : np.ndarray
        Source frequency points (Hz).
    source_values : np.ndarray
        Source values at source_freqs points.
    kind : str
        Interpolation kind (default "linear").

    Returns
    -------
    np.ndarray
        Interpolated values at target_freqs.
    """
    if len(source_freqs) < 2 or len(source_values) < 2:
        return np.full_like(target_freqs, np.nan, dtype=float)

    try:
        f_interp = interp1d(
            np.log10(source_freqs),
            source_values,
            kind=kind,
            bounds_error=False,
            fill_value=np.nan,
            assume_sorted=True,
        )
        return f_interp(np.log10(target_freqs))
    except (ValueError, np.Error):
        return np.full_like(target_freqs, np.nan, dtype=float)


def compare_spl(
    bem: BemReferenceData,
    tmm: TmmPredictionData,
    threshold_db: float = 3.0,
) -> BemValidationResult:
    """
    Compare TMM SPL prediction against BEM reference.

    Computes delta = TMM - BEM at each frequency point where both have data.

    Parameters
    ----------
    bem : BemReferenceData
        BEM reference data.
    tmm : TmmPredictionData
        TMM prediction data.
    threshold_db : float
        Maximum acceptable RMS error in dB. Default 3.0 dB.

    Returns
    -------
    BemValidationResult
        Validation result with delta statistics and pass/fail assessment.
    """
    if not bem.has_spl():
        raise ValueError("BEM reference has no SPL data")
    if tmm.spl is None or len(tmm.spl) == 0:
        raise ValueError("TMM prediction has no SPL data")

    common_lo = max(bem.freqs.min(), tmm.freqs.min())
    common_hi = min(bem.freqs.max(), tmm.freqs.max())

    bem_freqs_valid = bem.freqs[(bem.freqs >= common_lo) & (bem.freqs <= common_hi)]
    if len(bem_freqs_valid) < 3:
        raise ValueError(
            f"Insufficient overlapping frequency points: {len(bem_freqs_valid)}"
        )

    bem_spl_interp = interpolate_to_reference(bem_freqs_valid, bem.freqs, bem.spl)
    tmm_spl_interp = interpolate_to_reference(bem_freqs_valid, tmm.freqs, tmm.spl)

    valid = np.isfinite(bem_spl_interp) & np.isfinite(tmm_spl_interp)
    if valid.sum() < 3:
        raise ValueError(f"Insufficient valid interpolation points: {valid.sum()}")

    delta_spl = tmm_spl_interp[valid] - bem_spl_interp[valid]

    mean_delta = float(np.mean(delta_spl))
    std_delta = float(np.std(delta_spl))
    rmse = float(np.sqrt(np.mean(delta_spl**2)))
    max_abs = float(np.max(np.abs(delta_spl)))
    max_d = float(np.max(delta_spl))
    min_d = float(np.min(delta_spl))

    result = BemValidationResult(
        freqs=bem_freqs_valid[valid],
        delta_spl=delta_spl,
        mean_delta_db=mean_delta,
        std_delta_db=std_delta,
        rmse_db=rmse,
        max_abs_delta_db=max_abs,
        max_delta_db=max_d,
        min_delta_db=min_d,
        threshold_db=threshold_db,
        pass_threshold=rmse <= threshold_db,
    )

    if bem.has_impedance() and tmm.impedance is not None:
        bem_z_interp = interpolate_to_reference(
            bem_freqs_valid, bem.freqs, bem.impedance.real
        )
        bem_zi_interp = interpolate_to_reference(
            bem_freqs_valid, bem.freqs, bem.impedance.imag
        )
        tmm_z_interp = interpolate_to_reference(
            bem_freqs_valid, tmm.freqs, tmm.impedance.real
        )
        tmm_zi_interp = interpolate_to_reference(
            bem_freqs_valid, tmm.freqs, tmm.impedance.imag
        )

        dz_r = tmm_z_interp[valid] - bem_z_interp[valid]
        dz_i = tmm_zi_interp[valid] - bem_zi_interp[valid]
        z_rmse = float(np.sqrt(np.mean(dz_r**2 + dz_i**2)))

        result.delta_impedance_real = dz_r
        result.delta_impedance_imag = dz_i
        result.impedance_rmse = z_rmse

    if bem.has_directivity() and tmm.directivity_horizontal is not None:
        if (
            tmm.directivity_angles is not None
            and bem.directivity_angles is not None
        ):
            common_angles = np.intersect1d(
                tmm.directivity_angles, bem.directivity_angles
            )
            if len(common_angles) > 0:
                tmm_idx = [
                    np.argmin(np.abs(tmm.directivity_angles - a))
                    for a in common_angles
                ]
                bem_idx = [
                    np.argmin(np.abs(bem.directivity_angles - a))
                    for a in common_angles
                ]

                tmm_dh = tmm.directivity_horizontal[tmm_idx]
                bem_dh = bem.directivity_horizontal[bem_idx]
                result.delta_directivity_horizontal = tmm_dh - bem_dh

                if tmm.directivity_vertical is not None:
                    tmm_dv = tmm.directivity_vertical[tmm_idx]
                    bem_dv = bem.directivity_vertical[bem_idx]
                    result.delta_directivity_vertical = tmm_dv - bem_dv

                dd = np.concatenate([result.delta_directivity_horizontal])
                if result.delta_directivity_vertical is not None:
                    dd = np.concatenate([dd, result.delta_directivity_vertical])
                result.directivity_rmse = float(np.sqrt(np.mean(dd**2)))

    return result


def compare_impedance(
    bem: BemReferenceData,
    tmm: TmmPredictionData,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compare TMM impedance prediction against BEM reference.

    Parameters
    ----------
    bem : BemReferenceData
        BEM reference data.
    tmm : TmmPredictionData
        TMM prediction data.

    Returns
    -------
    tuple of (delta_real, delta_imag, rmse)
        Difference in real and imaginary parts, and RMSE in ohms.
    """
    if not bem.has_impedance():
        raise ValueError("BEM reference has no impedance data")
    if tmm.impedance is None:
        raise ValueError("TMM prediction has no impedance data")

    common_lo = max(bem.freqs.min(), tmm.freqs.min())
    common_hi = min(bem.freqs.max(), tmm.freqs.max())
    bem_freqs_valid = bem.freqs[(bem.freqs >= common_lo) & (bem.freqs <= common_hi)]

    bem_z_interp = interpolate_to_reference(bem_freqs_valid, bem.freqs, bem.impedance.real)
    bem_zi_interp = interpolate_to_reference(
        bem_freqs_valid, bem.freqs, bem.impedance.imag
    )
    tmm_z_interp = interpolate_to_reference(
        bem_freqs_valid, tmm.freqs, tmm.impedance.real
    )
    tmm_zi_interp = interpolate_to_reference(
        bem_freqs_valid, tmm.freqs, tmm.impedance.imag
    )

    valid = np.isfinite(bem_z_interp) & np.isfinite(tmm_z_interp)
    delta_real = tmm_z_interp[valid] - bem_z_interp[valid]
    delta_imag = tmm_zi_interp[valid] - bem_zi_interp[valid]
    rmse = float(np.sqrt(np.mean(delta_real**2 + delta_imag**2)))

    return delta_real, delta_imag, rmse


def assess_tmm_validity(
    result: BemValidationResult,
    thresholds: dict = None,
) -> dict:
    """
    Assess TMM validity based on BEM comparison results.

    Parameters
    ----------
    result : BemValidationResult
        BEM comparison result.
    thresholds : dict, optional
        Custom threshold values with keys: rmse_db, max_abs_db, mean_db, std_db.

    Returns
    -------
    dict
        Assessment with pass/fail for each criterion and overall verdict.
    """
    if thresholds is None:
        thresholds = {
            "rmse_db": 3.0,
            "max_abs_db": 6.0,
            "mean_db": 2.0,
            "std_db": 2.0,
        }

    checks = {
        "rmse": result.rmse_db <= thresholds.get("rmse_db", 3.0),
        "max_abs": result.max_abs_delta_db <= thresholds.get("max_abs_db", 6.0),
        "mean": abs(result.mean_delta_db) <= thresholds.get("mean_db", 2.0),
        "std": result.std_delta_db <= thresholds.get("std_db", 2.0),
    }

    if result.impedance_rmse is not None:
        checks["impedance"] = result.impedance_rmse <= thresholds.get(
            "impedance_ohm", 2.0
        )

    if result.directivity_rmse is not None:
        checks["directivity"] = result.directivity_rmse <= thresholds.get(
            "directivity_db", 3.0
        )

    overall_pass = all(checks.values())

    return {
        "checks": checks,
        "overall_pass": overall_pass,
        "rmse_db": result.rmse_db,
        "max_abs_delta_db": result.max_abs_delta_db,
        "mean_delta_db": result.mean_delta_db,
        "std_delta_db": result.std_delta_db,
        "thresholds": thresholds,
    }


def generate_standard_horn_reference(
    geometry_type: Literal["exponential", "conical", "hyperbolic", "tractrix"],
    throat_area_m2: float,
    mouth_area_m2: float,
    path_length_m: float,
    n_segments: int = 200,
    output_path: Path | str | None = None,
) -> dict:
    """
    Generate reference geometry parameters for BEM validation.

    Creates a standardized horn geometry that can be used to:
    1. Generate a mesh for BEM simulation (e.g. via gmsh)
    2. Run pyhorn TMM simulation

    Parameters
    ----------
    geometry_type : str
        Horn profile type: "exponential", "conical", "hyperbolic", "tractrix".
    throat_area_m2 : float
        Throat cross-sectional area in m².
    mouth_area_m2 : float
        Mouth cross-sectional area in m².
    path_length_m : float
        Axial path length in m.
    n_segments : int
        Number of segments for TMM discretization.
    output_path : Path or str, optional
        If provided, write the geometry JSON to this path.

    Returns
    -------
    dict
        Geometry parameters including profile function and discretized segments.
    """
    import math

    if throat_area_m2 <= 0 or mouth_area_m2 <= 0:
        raise ValueError("Throat and mouth areas must be positive")
    if path_length_m <= 0:
        raise ValueError("Path length must be positive")

    a_throat = math.sqrt(throat_area_m2 / math.pi)
    a_mouth = math.sqrt(mouth_area_m2 / math.pi)

    geometry = {
        "geometry_type": geometry_type,
        "throat_area_m2": throat_area_m2,
        "mouth_area_m2": mouth_area_m2,
        "path_length_m": path_length_m,
        "throat_radius_m": a_throat,
        "mouth_radius_m": a_mouth,
        "n_segments": n_segments,
        "mouth_radiation": "levine",
        "ang": math.pi,
        "profile_params": {},
    }

    if geometry_type == "exponential":
        m = math.log(mouth_area_m2 / throat_area_m2) / path_length_m
        geometry["profile_params"] = {
            "m": m,
            "throat_radius": a_throat,
            "mouth_radius": a_mouth,
        }
        geometry["description"] = (
            f"Exponential horn: S(x) = S0 * exp(m*x), "
            f"m = {m:.4f} m^-1"
        )

    elif geometry_type == "conical":
        geometry["profile_params"] = {
            "throat_radius": a_throat,
            "mouth_radius": a_mouth,
            "length": path_length_m,
        }
        geometry["description"] = (
            f"Conical horn: S(x) = S0 * (1 + x/L), linear taper"
        )

    elif geometry_type == "hyperbolic":
        t = 0.7
        geometry["profile_params"] = {"t": t}
        geometry["description"] = (
            f"Hyperbolic horn: S(x) = S0 * sinh(t*x/L) / sinh(t), t = {t}"
        )

    elif geometry_type == "tractrix":
        geometry["description"] = (
            "Tractrix horn: dS/S = -dx/sqrt(a^2 - x^2)"
        )

    if output_path is not None:
        with open(Path(output_path), "w") as f:
            json.dump(geometry, f, indent=2)

    return geometry
