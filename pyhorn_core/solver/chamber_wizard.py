"""Chamber Design Wizard — compute recommended Vrc, Lrc, Vtc, Atc, Ap1, Lpt from T-S parameters."""

import re
from dataclasses import dataclass
from typing import Optional


# ── Constants ──────────────────────────────────────────────────────────────────

# Target Qts for rear-chamber alignment (typical range 0.5–0.7)
QTS_TARGET = 0.6

# Default driver diameter fallback (58 mm) used when Sd is not in the YAML
DEFAULT_DRIVER_DIAM_M = 0.058

# Default Ap1 aperture diameter (50 mm)
AP1_DIAM_M = 0.050

# Default baffle / neck thickness (12 mm)
LPT_M = 0.012

# Throat chamber volume as fraction of Vas (0.2 %)
VTC_FRACTION_OF_VAS = 0.002


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class TSPParams:
    """Thiele-Small parameters parsed from a driver YAML."""
    fs: Optional[float]      # Hz
    qts: Optional[float]     # (-)
    vas: Optional[float]     # m³
    sd: Optional[float]     # m²
    sd_cm2: Optional[float] # cm²


@dataclass
class ComputedChamberParams:
    """Chamber geometry parameters computed from T-S parameters."""
    vrc_L: float   # rear chamber volume (litres)
    lrc_m: float   # rear chamber path length (metres)
    vtc_m3: float  # throat chamber volume (m³)
    vtc_cm3: float # throat chamber volume (cm³)
    atc_m2: float  # throat chamber area (m²)
    atc_cm2: float # throat chamber area (cm²)
    ap1_m2: float  # baffle aperture area (m²)
    ap1_cm2: float # baffle aperture area (cm²)
    lpt_m: float   # baffle thickness (m)
    lpt_cm: float  # baffle thickness (cm)


@dataclass
class Validation:
    """Validation warnings for chamber parameters."""
    vrc_warning: Optional[str]
    lrc_warning: Optional[str]
    vtc_warning: Optional[str]
    atc_warning: Optional[str]
    ap1_warning: Optional[str]
    lpt_warning: Optional[str]


# ── YAML parsing ───────────────────────────────────────────────────────────────

def parse_driver_param(yaml: str, key: str) -> Optional[float]:
    """Extract a numeric value from YAML by key name (case-insensitive, multiline)."""
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*:\s*([\d.e+-]+)", re.MULTILINE | re.IGNORECASE)
    m = pattern.search(yaml)
    return float(m.group(1)) if m else None


def parse_tsp(yaml: str) -> TSPParams:
    """Parse Thiele-Small parameters from a driver YAML string."""
    sd_m2 = parse_driver_param(yaml, "sd")
    sd_cm2 = sd_m2 * 1e4 if sd_m2 is not None else None
    return TSPParams(
        fs=parse_driver_param(yaml, "fs"),
        qts=parse_driver_param(yaml, "qts"),
        vas=parse_driver_param(yaml, "vas"),
        sd=sd_m2,
        sd_cm2=sd_cm2,
    )


# ── Core computation ──────────────────────────────────────────────────────────

def compute_chamber_params(tsp: TSPParams) -> ComputedChamberParams:
    """Compute recommended chamber geometry from T-S parameters.

    Formulas (matching POST /chamber-wizard/compute API endpoint):
      Vrc  = Vas × (Qts² / Qts_target² − 1)   [m³]
             Positive when Qts > Qts_target; for low-Qts drivers,
             Vrc will be 0 (add a rear chamber only makes sense when
             the driver Q is already higher than desired).
      Lrc  = Vrc / Atc                          [m]
      Atc  = Sd  (throat chamber area ≈ driver piston area)
      Vtc  = 0.002 × Vas                        [m³]  (0.2 % of Vas)
      Ap1  = π × (50mm/2)²  (fixed 50 mm baffle aperture)
      Lpt  = 12 mm  (fixed baffle thickness)
    """
    import math

    # Atc ≈ Sd (driver piston area), fallback to default diameter
    if tsp.sd is not None:
        atc_m2 = tsp.sd
    else:
        atc_m2 = (math.pi / 4.0) * (DEFAULT_DRIVER_DIAM_M ** 2)
    atc_cm2 = atc_m2 * 1e4

    # Vrc = Vas × (Qts² / Qts_target² − 1)  [m³]
    if tsp.vas is not None and tsp.qts is not None and tsp.qts > 0:
        ratio = (tsp.qts ** 2) / (QTS_TARGET ** 2)
        if ratio > 1.0:
            vrc_m3 = tsp.vas * (ratio - 1.0)
        else:
            vrc_m3 = 0.0  # Qts ≤ Qts_target: no rear chamber needed
    else:
        vrc_m3 = 0.0

    # Lrc = Vrc / Atc  [m]
    if vrc_m3 > 0 and atc_m2 > 0:
        lrc_m = vrc_m3 / atc_m2
    else:
        lrc_m = 0.0

    # Vtc = 0.002 × Vas  [m³]
    vtc_m3 = (0.002 * tsp.vas) if tsp.vas is not None else 0.0
    vtc_cm3 = vtc_m3 * 1e6

    # Ap1 = fixed 50 mm diameter hole
    ap1_m2 = (math.pi / 4.0) * (AP1_DIAM_M ** 2)
    ap1_cm2 = ap1_m2 * 1e4

    # Lpt = fixed 12 mm
    lpt_m = LPT_M
    lpt_cm = lpt_m * 100.0

    return ComputedChamberParams(
        vrc_L=vrc_m3 * 1000.0,
        lrc_m=lrc_m,
        vtc_m3=vtc_m3,
        vtc_cm3=vtc_cm3,
        atc_m2=atc_m2,
        atc_cm2=atc_cm2,
        ap1_m2=ap1_m2,
        ap1_cm2=ap1_cm2,
        lpt_m=lpt_m,
        lpt_cm=lpt_cm,
    )


# ── Validation ─────────────────────────────────────────────────────────────────

def validate_chamber(
    p: ComputedChamberParams,
    tsp: TSPParams,
    qts_target: float = QTS_TARGET,
) -> Validation:
    """Return warnings for out-of-range or physically suspicious values.

    Parameters
    ----------
    p : ComputedChamberParams
        Computed chamber geometry values.
    tsp : TSPParams
        Driver Thiele-Small parameters.
    qts_target : float
        Target Qts used for the alignment (default: QTS_TARGET = 0.6).
        When Qts ≤ qts_target, Vrc=0 is intentional (no rear chamber needed),
        so the "too small" warning is suppressed.
    """
    warnings = Validation(
        vrc_warning=None,
        lrc_warning=None,
        vtc_warning=None,
        atc_warning=None,
        ap1_warning=None,
        lpt_warning=None,
    )

    # Vrc: 0.5–30 L is a practical range for a BLH rear chamber
    # Vrc=0 is intentional when Qts ≤ qts_target (driver Q is already low enough)
    if p.vrc_L < 0.5:
        if not (tsp.qts is not None and tsp.qts <= qts_target):
            warnings.vrc_warning = "too small — system Q too high, peaky response"
    elif p.vrc_L > 30:
        warnings.vrc_warning = "very large — may be unnecessary"

    # Lrc: 3–80 cm is a realistic range for a box internal dimension
    # Lrc=0 is intentional when Qts ≤ qts_target (no rear chamber needed)
    lrc_cm = p.lrc_m * 100.0
    if lrc_cm < 3.0 or lrc_cm > 80.0:
        if not (tsp.qts is not None and tsp.qts <= qts_target and p.vrc_L == 0.0):
            warnings.lrc_warning = "unrealistic for a box dimension"

    # Vtc: 5–500 cm³ is a practical range for a throat chamber
    if p.vtc_cm3 < 5.0:
        warnings.vtc_warning = "tiny — dust cap clearance only"
    elif p.vtc_cm3 > 500.0:
        warnings.vtc_warning = "unusually large for a throat chamber"

    # Atc: should be comparable to Sd (0.5× to 2×)
    if tsp.sd_cm2 is not None:
        if p.atc_cm2 > 2.0 * tsp.sd_cm2:
            warnings.atc_warning = "Atc > 2× Sd — weak coupling, reflections"
        elif p.atc_cm2 < 0.5 * tsp.sd_cm2:
            warnings.atc_warning = "Atc < 0.5× Sd — restricted airflow"

    # Ap1: should be at least 0.5× Sd
    if tsp.sd_cm2 is not None and p.ap1_cm2 < 0.5 * tsp.sd_cm2:
        warnings.ap1_warning = "Ap1 < 0.5× Sd — restricted airflow"

    # Lpt: 3–50 cm baffle thickness
    if p.lpt_cm < 3.0 or p.lpt_cm > 50.0:
        warnings.lpt_warning = "unusual baffle thickness"

    return warnings


# ── YAML output ────────────────────────────────────────────────────────────────

def build_yaml_snippet(p: ComputedChamberParams) -> str:
    """Generate a YAML snippet for project files."""
    return (
        f"rear_chamber:\n"
        f"  vrc: {p.vrc_L:.4f}\n"
        f"  lrc: {(p.lrc_m * 100.0):.2f}\n"
        f"throat_chamber:\n"
        f"  vtc: {p.vtc_m3:.6f}\n"
        f"  atc: {p.atc_cm2:.4f}\n"
        f"throat_adapter:\n"
        f"  ap1: {p.ap1_cm2:.4f}\n"
        f"  lpt: {(p.lpt_m * 1000.0):.2f}"
    )
