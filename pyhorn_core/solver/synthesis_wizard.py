"""Horn System Synthesis Wizard — pyhorn_core solver.

Full end-to-end synthesis from T-S parameters + frequency targets
→ complete horn system (driver, horn geometry, chambers, baffle).

Reference: Hornresp manual page 067 "System Design / Hornresp Synthesis Wizard".

This module provides the core solver-only implementation.
The API endpoint and CLI commands are separate layers (pyhorn_api / pyhorn_cli).

Inputs (SynthesisInput)
  - T-S parameters: fs, qts, vas, sd, re, bl, mms, cms, rms, qes, qms, le
  - Frequency targets: f3_target_hz (lower -3dB cutoff), f7_target_hz (upper -3dB)
  - System type: FLH | BLH
  - Room gain compensation toggle
  - Preferred enclosure dimensions (W × H × D in metres, optional)

Outputs (SynthesisOutput)
  - HornGeometry YAML-ready dict
  - DriverSpecs YAML-ready dict
  - Validation warnings list
  - Explanation strings per computed value

Synthesis approach
------------------
Step 1 — F12 (horn cutoff): Set F12 = f3_target / 1.2  (typical BLH scaling factor,
         empirical — Hornresp default; tighter horns can use 0.8–1.0)

Step 2 — S1 (throat area): S1 = Atc ≈ Sd  (throat chamber area ≈ driver piston area;
         matches Hornresp System Design default)

Step 3 — S2 (mouth area): Use catenoidal horn formula:
         S2 = S1 × (1 + (2π·F12·L12/c)²)
         Rearranged: L12 = (c/2π·F12) × √(S2/S1 − 1)
         Given MDF constraints (2400 × 1200 mm sheet), sweep L12 from 0.5–2.0 m
         and pick S2 that fits the cabinet.

Step 4 — Horn profile: Use catenoidal (T=1) for smoothest expansion;
         optionally split into sections: straight_throat → exponential/main → mouth

Step 5 — Rear chamber (Vrc, Lrc): Vrc = Vas × (Qts²/Qts_target² − 1)
         Target Qts_alignment = 0.55 (good for BLH; allows low f3 without Q spike)
         Lrc = Vrc / Atc  (back-calculate from box dimensions if dims given)

Step 6 — Throat chamber (Vtc, Atc): Vtc ≈ 0.002 × Vas  (0.2% of Vas, Hornresp default)
         Atc ≈ Sd

Step 7 — Throat adapter (Ap1, Lpt): Ap1 = min(1.1×Sd, π·(50mm)²)
         Lpt = 12 mm (default baffle thickness)

Step 8 — Baffle: W_baffle = max(driver_dim + 20mm margin, width_constraint)
         Baffle area for radiation: Abaffle ≥ 3×Sd

Units: All internal computation in SI (m², m, Hz).  YAML I/O handles unit conversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import math

# ─── Physical constants ───────────────────────────────────────────────────────

C_SOUND = 343.0  # m/s — speed of sound at 20 °C
RHO = 1.21       # kg/m³ — air density


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class SynthesisInput:
    """Inputs to the Horn System Synthesis Wizard."""

    # ── T-S parameters ───────────────────────────────────────────────────────
    fs: float       # Hz — driver free-air resonance
    qts: float      # (—) — total Q atfs
    vas: float      # m³ — equivalent compliance volume
    sd: float       # m² — piston radiating area
    re: float       # Ω — voice coil resistance
    bl: float       # N/A — force factor
    mms: float      # kg — moving mass
    cms: float      # m/N — driver compliance
    rms: float      # kg/s — mechanical resistance
    qes: float      # (—) — electrical Q
    qms: float      # (—) — mechanical Q
    le: float = 0.0 # H — voice coil inductance (optional)

    # ── Frequency targets ───────────────────────────────────────────────────
    f3_target_hz: float = 50.0   # Hz — desired -3 dB low-frequency cutoff
    f7_target_hz: float = 8000.0 # Hz — desired -3 dB high-frequency cutoff

    # ── System configuration ────────────────────────────────────────────────
    system_type: str = "BLH"    # FLH | BLH
    enclosure_type: str = "FLH" # FLH | BLH  (alias for system_type)

    # ── Target alignment Q ─────────────────────────────────────────────────
    qts_alignment: float = 0.55  # (—) — target Qts for the rear-chamber alignment

    # ── Physical constraints ───────────────────────────────────────────────
    max_mouth_area_m2: float = 0.12   # m² — 2400×1200 mm MDF sheet mouth budget
    max_path_length_m: float = 2.0    # m — maximum horn path length
    baffle_thickness_m: float = 0.012 # m — 12 mm MDF default

    # ── Room gain compensation ─────────────────────────────────────────────
    room_gain_db: float = 0.0  # dB — manual room gain offset

    @property
    def f12_hz(self) -> float:
        """Horn cutoff frequency: F12 ≈ f3 / 1.2 (empirical BLH scaling)."""
        return self.f3_target_hz / 1.2

    def __post_init__(self) -> None:
        # Normalise system_type / enclosure_type
        if self.system_type not in ("FLH", "BLH"):
            self.system_type = "BLH"
        self.enclosure_type = self.system_type


@dataclass
class HornSectionSpec:
    """Specification for one horn section."""
    name: str
    profile_type: str       # straight | exponential | hyperbolic | catenoidal
    length_m: float         # metres
    start_area_m2: float    # m²
    end_area_m2: float      # m²
    hyperbolic_t: Optional[float] = None


@dataclass
class HornSpec:
    """Synthesised horn geometry as a list of sections."""
    sections: list[HornSectionSpec]
    throat_area_m2: float   # S1
    mouth_area_m2: float    # S2
    path_length_m: float    # total
    f12_computed_hz: float  # actual cutoff achieved


@dataclass
class ChamberSpec:
    """Synthesised chamber parameters."""
    vrc_m3: float    # rear chamber volume (m³)
    vrc_L: float     # rear chamber volume (litres)
    lrc_m: float     # rear chamber path length (m)
    vtc_m3: float    # throat chamber volume (m³)
    atc_m2: float    # throat chamber area (m²)
    ap1_m2: float    # baffle aperture area (m²)
    lpt_m: float     # baffle thickness / neck length (m)
    explanation: dict[str, str] = field(default_factory=dict)


@dataclass
class ValidationWarning:
    """A validation warning with severity and message."""
    field: str
    severity: str   # INFO | WARN | ERROR
    message: str


@dataclass
class SynthesisOutput:
    """Complete output of the Horn System Synthesis Wizard."""
    horn: HornSpec
    chambers: ChamberSpec
    driver_sd_m2: float
    baffle_area_m2: float
    warnings: list[ValidationWarning]
    explanation: dict[str, str] = field(default_factory=dict)


# ─── Core synthesis ───────────────────────────────────────────────────────────

def _area_ratio_from_f12_l12(f12_hz: float, l12_m: float, s1_m2: float) -> float:
    """Return the mouth-to-throat area ratio for a catenoidal horn.

    R = S2/S1 = cosh²(u_total),  u_total = 2π·F12·L12 / c
    Rearranged: R = cosh²(2π·F12·L12 / c)
    """
    u_total = 2.0 * math.pi * f12_hz * l12_m / C_SOUND
    return math.cosh(u_total) ** 2


def _solve_horn_path_length(
    s1_m2: float,
    s2_m2: float,
    f12_hz: float,
    max_l_m: float = 2.0,
) -> tuple[float, float]:
    """Solve for horn path length that gives S2 from S1 at F12.

    Returns (L12_m, actual_F12_hz).
    If S2 is too large for the max length, returns (max_l_m, computed_f12).
    """
    if s2_m2 <= s1_m2:
        raise ValueError(f"S2 ({s2_m2*1e4:.2f} cm²) must be > S1 ({s1_m2*1e4:.2f} cm²)")
    ratio = s2_m2 / s1_m2
    if ratio <= 1.0 + 1e-12:
        raise ValueError("S2 must be strictly greater than S1 for an expanding horn")
    u_total = math.acosh(math.sqrt(ratio))
    l_needed = u_total * C_SOUND / (2.0 * math.pi * f12_hz)
    actual_f12 = f12_hz
    if l_needed > max_l_m:
        # Mouth area is too large for the constraint; use max length and compute actual F12
        l_needed = max_l_m
        actual_f12 = C_SOUND / (2.0 * math.pi * l_needed) * math.sqrt(ratio - 1.0)
    return l_needed, actual_f12


def synthesize_horn_system(inp: SynthesisInput) -> SynthesisOutput:
    """Synthesise a complete horn system from T-S parameters and frequency targets.

    Parameters
    ----------
    inp : SynthesisInput
        T-S parameters + frequency targets + physical constraints.

    Returns
    -------
    SynthesisOutput
        Complete horn geometry (as sections), chamber parameters, baffle area,
        and validation warnings.

    Algorithm
    ---------
    1. S1 (throat) = Atc ≈ Sd  (Hornresp System Design default)
    2. F12 = f3_target / 1.2   (empirical BLH scaling factor)
    3. Sweep path length L12 from 0.3–max_path_length_m; pick S2 that fits within
       max_mouth_area_m2 and gives smooth catenoidal expansion.
    4. If S2 > max_mouth_area: use max_mouth_area as constraint, recompute L12.
    5. Rear chamber: Vrc = Vas × (Qts² / Qts_alignment² − 1); Lrc = Vrc / Atc
    6. Throat chamber: Vtc = 0.002 × Vas; Atc = Sd
    7. Throat adapter: Ap1 = min(1.1×Sd, π·(50mm)²); Lpt = baffle_thickness_m
    8. Baffle area: ≥ 3 × Sd
    9. Warn if L12 > 1.5 m (MDF sheet limit), if Vrc is extreme, etc.
    """
    warnings: list[ValidationWarning] = []
    explanation: dict[str, str] = {}

    # ── Step 1: Throat area / throat chamber ────────────────────────────────
    atc_m2 = inp.sd
    vtc_m3 = 0.002 * inp.vas
    explanation["atc"] = f"Atc = Sd = {inp.sd*1e4:.2f} cm² (Hornresp default)"
    explanation["vtc"] = f"Vtc = 0.2% of Vas = {vtc_m3*1e6:.2f} cm³"

    # ── Step 2: Horn cutoff ─────────────────────────────────────────────────
    f3 = inp.f3_target_hz
    f12 = inp.f12_hz
    explanation["f12"] = f"F12 = f3/1.2 = {f3:.1f}/1.2 = {f12:.1f} Hz (empirical BLH scaling)"

    # ── Step 3: Throat area S1 = Atc ────────────────────────────────────────
    s1_m2 = atc_m2
    explanation["s1"] = f"S1 = Atc = {s1_m2*1e4:.4f} cm²"

    # ── Step 4: Solve for path length and mouth area ────────────────────────
    #
    # Strategy: Start with desired mouth area budget (max_mouth_area_m2),
    # solve for required L12. If L12 > max_path_length_m, reduce mouth area
    # until L12 fits.
    target_s2_m2 = inp.max_mouth_area_m2
    l12_m, actual_f12 = _solve_horn_path_length(
        s1_m2, target_s2_m2, f12, inp.max_path_length_m
    )

    # If L12 came out > max_path_length_m, the mouth area is too large for the
    # horn to expand that quickly at F12. Reduce S2 iteratively.
    if l12_m > inp.max_path_length_m:
        # Binary search for the largest S2 that fits in max_path_length_m
        s2_lo, s2_hi = s1_m2 * 1.01, target_s2_m2
        for _ in range(30):  # 30 iterations → ~1e-9 precision
            s2_mid = (s2_lo + s2_hi) / 2.0
            l_try, _ = _solve_horn_path_length(s1_m2, s2_mid, f12, inp.max_path_length_m)
            if l_try <= inp.max_path_length_m:
                s2_lo = s2_mid
            else:
                s2_hi = s2_mid
        l12_m = inp.max_path_length_m
        actual_f12 = f12  # Keep target F12; L12 is the constraint
        target_s2_m2 = s2_lo

    l12_m, actual_f12 = _solve_horn_path_length(
        s1_m2, target_s2_m2, f12, inp.max_path_length_m
    )
    s2_m2 = target_s2_m2

    explanation["l12"] = (
        f"L12 = {l12_m:.4f} m  "
        f"(F12={actual_f12:.1f} Hz, S2={s2_m2*1e4:.2f} cm²)"
    )

    # ── Step 5: Section decomposition ───────────────────────────────────────
    # Straight throat section (first 0.1–0.15 m) → reduces throat resonance
    throat_section_len = min(0.15, l12_m * 0.1)
    throat_end_area = s1_m2  # Straight section: constant area
    main_len = l12_m - throat_section_len
    main_end_area = s2_m2

    sections: list[HornSectionSpec] = [
        HornSectionSpec(
            name="throat",
            profile_type="straight",
            length_m=throat_section_len,
            start_area_m2=s1_m2,
            end_area_m2=throat_end_area,
        ),
    ]

    # Main flare: catenoidal (T=1) gives smoothest expansion
    if main_len > 0.05:
        sections.append(
            HornSectionSpec(
                name="main_horn",
                profile_type="catenoidal",
                length_m=main_len,
                start_area_m2=s1_m2,
                end_area_m2=main_end_area,
                hyperbolic_t=1.0,
            )
        )
    else:
        # Very short horn: just one exponential section
        sections[0] = HornSectionSpec(
            name="horn",
            profile_type="catenoidal",
            length_m=l12_m,
            start_area_m2=s1_m2,
            end_area_m2=s2_m2,
            hyperbolic_t=1.0,
        )

    horn = HornSpec(
        sections=sections,
        throat_area_m2=s1_m2,
        mouth_area_m2=s2_m2,
        path_length_m=l12_m,
        f12_computed_hz=actual_f12,
    )

    # ── Step 6: Rear chamber ─────────────────────────────────────────────────
    if inp.qts_alignment <= 0:
        vrc_m3 = 0.0
        warnings.append(ValidationWarning(
            field="vrc",
            severity="WARN",
            message=(
                f"qts_alignment={inp.qts_alignment:.3f} is invalid (must be > 0): "
                "no rear chamber will be synthesised"
            ),
        ))
    else:
        ratio = (inp.qts ** 2) / (inp.qts_alignment ** 2)
        if ratio > 1.0:
            vrc_m3 = inp.vas * (ratio - 1.0)
        else:
            vrc_m3 = 0.0
            warnings.append(ValidationWarning(
                field="vrc",
                severity="INFO",
                message=(
                    f"Qts={inp.qts:.3f} ≤ Qts_alignment={inp.qts_alignment:.3f}: "
                    "no rear chamber needed — driver Q is already low enough"
                ),
            ))
    lrc_m = (vrc_m3 / atc_m2) if atc_m2 > 0 else 0.0

    explanation["vrc"] = (
        f"Vrc = Vas × (Qts²/Qts_align² − 1) = {inp.vas*1e6:.2f} × "
        f"({inp.qts:.3f}²/{inp.qts_alignment:.3f}² − 1) = {vrc_m3*1e3:.2f} L"
    )
    explanation["lrc"] = f"Lrc = Vrc / Atc = {lrc_m*100:.2f} cm"

    # ── Step 7: Throat adapter ───────────────────────────────────────────────
    ap1_m2 = min(inp.sd * 1.1, math.pi * (0.050 ** 2) / 4.0)  # 1.1×Sd or 50mm hole
    lpt_m = inp.baffle_thickness_m
    explanation["ap1"] = (
        f"Ap1 = min(1.1×Sd, π·(50mm)²) = {ap1_m2*1e4:.2f} cm²"
    )
    explanation["lpt"] = f"Lpt = {lpt_m*1000:.1f} mm baffle thickness"

    chambers = ChamberSpec(
        vrc_m3=vrc_m3,
        vrc_L=vrc_m3 * 1000.0,
        lrc_m=lrc_m,
        vtc_m3=vtc_m3,
        atc_m2=atc_m2,
        ap1_m2=ap1_m2,
        lpt_m=lpt_m,
        explanation=explanation,
    )

    # ── Step 8: Baffle area ──────────────────────────────────────────────────
    baffle_area_m2 = max(3.0 * inp.sd, 0.02)  # ≥ 3× Sd or 200 cm² minimum
    explanation["baffle"] = f"Abaffle = max(3×Sd, 200 cm²) = {baffle_area_m2*1e4:.0f} cm²"

    # ── Step 9: Validation warnings ──────────────────────────────────────────
    if l12_m > 1.5:
        warnings.append(ValidationWarning(
            field="l12",
            severity="WARN",
            message=(
                f"L12={l12_m:.2f} m exceeds 1.5 m — "
                "verify it fits inside your 2400 mm MDF sheet height"
            ),
        ))
    if vrc_m3 * 1000 > 30.0:
        warnings.append(ValidationWarning(
            field="vrc",
            severity="WARN",
            message=f"Vrc={vrc_m3*1000:.1f} L is large — verify box dimensions are realistic",
        ))
    if vrc_m3 * 1000 < 0.5 and vrc_m3 > 0:
        warnings.append(ValidationWarning(
            field="vrc",
            severity="WARN",
            message="Vrc < 0.5 L — very small rear chamber, may cause alignment issues",
        ))
    if actual_f12 > f12 * 1.05:
        warnings.append(ValidationWarning(
            field="f12",
            severity="WARN",
            message=(
                f"Actual F12={actual_f12:.1f} Hz exceeds target {f12:.1f} Hz by >5% "
                "(MDF path-length constraint). Consider allowing a larger mouth area."
            ),
        ))
    if inp.f7_target_hz < 5000:
        warnings.append(ValidationWarning(
            field="f7",
            severity="INFO",
            message=(
                f"f7={inp.f7_target_hz:.0f} Hz is low — "
                "upper cutoff may be limited by horn's natural compression ratio"
            ),
        ))

    output = SynthesisOutput(
        horn=horn,
        chambers=chambers,
        driver_sd_m2=inp.sd,
        baffle_area_m2=baffle_area_m2,
        warnings=warnings,
        explanation=explanation,
    )
    return output


# ─── YAML serialisation ───────────────────────────────────────────────────────

def synthesis_to_horn_geometry_yaml(syn: SynthesisOutput) -> str:
    """Serialise SynthesisOutput to a YAML string suitable for pyhorn horn geometry."""
    lines = [
        "# ──────────────────────────────────────────────────────────────────────────",
        "#  Synthesised Horn Geometry  —  generated by HornSynthesisWizard",
        "# ──────────────────────────────────────────────────────────────────────────",
        "#",
        f"#  Throat area (S1):  {syn.horn.throat_area_m2*1e4:.4f} cm²",
        f"#  Mouth area (S2):   {syn.horn.mouth_area_m2*1e4:.2f} cm²",
        f"#  Path length:       {syn.horn.path_length_m:.4f} m",
        f"#  Cutoff (F12):      {syn.horn.f12_computed_hz:.2f} Hz",
        "#",
        "sections:",
    ]
    for sec in syn.horn.sections:
        profile = sec.profile_type
        indent = "  "
        lines.append(f"{indent}- name: {sec.name}")
        lines.append(f"{indent}  profile_type: {profile}")
        lines.append(f"{indent}  length: {sec.length_m:.4f}")
        lines.append(f"{indent}  start_area: {sec.start_area_m2:.6f}")
        lines.append(f"{indent}  end_area: {sec.end_area_m2:.6f}")
        # catenoidal is hyperbolic with T=1 — output hyperbolic_t so solver accepts it
        if sec.profile_type == "catenoidal" and sec.hyperbolic_t is not None:
            lines.append(f"{indent}  hyperbolic_t: {sec.hyperbolic_t:.3f}")
        elif sec.profile_type == "hyperbolic" and sec.hyperbolic_t is not None:
            lines.append(f"{indent}  hyperbolic_t: {sec.hyperbolic_t:.3f}")
    lines.append("")
    lines.append("# ── Rear chamber ─────────────────────────────────────────────────────────")
    lines.append(f"vrc: {syn.chambers.vrc_L:.4f}   # litres")
    lines.append(f"lrc: {syn.chambers.lrc_m*100:.2f}   # cm")
    lines.append("")
    lines.append("# ── Throat chamber ──────────────────────────────────────────────────────")
    lines.append(f"vtc: {syn.chambers.vtc_m3:.6f}   # m³")
    lines.append(f"atc: {syn.chambers.atc_m2*1e4:.4f}   # cm²")
    lines.append("")
    lines.append("# ── Throat adapter ─────────────────────────────────────────────────────")
    lines.append(f"ap1: {syn.chambers.ap1_m2*1e4:.4f}   # cm²")
    lines.append(f"lpt: {syn.chambers.lpt_m*1000:.1f}   # mm")
    lines.append("")
    lines.append("# ── Radiation ──────────────────────────────────────────────────────────")
    lines.append(f"ang: 6.283185307   # 2π sr — half-space radiation")
    return "\n".join(lines)


def synthesis_to_driver_yaml(syn: SynthesisOutput, tsp: SynthesisInput) -> str:
    """Serialise T-S parameters to a driver YAML string."""
    return (
        "# ──────────────────────────────────────────────────────────────────────────\n"
        "#  Synthesised Driver Specs  —  generated by HornSynthesisWizard\n"
        "# ──────────────────────────────────────────────────────────────────────────\n"
        f"#  fs:   {tsp.fs:.2f} Hz\n"
        f"#  qts:  {tsp.qts:.4f}\n"
        f"#  qes:  {tsp.qes:.4f}\n"
        f"#  qms:  {tsp.qms:.4f}\n"
        f"#  vas:  {tsp.vas*1e6:.2f} cm³\n"
        f"#  re:   {tsp.re:.2f} Ω\n"
        f"#  bl:   {tsp.bl:.4f} N/A\n"
        f"#  mms:  {tsp.mms*1000:.4f} g\n"
        f"#  cms:  {tsp.cms*1e3:.6f} mm/N\n"
        f"#  sd:   {tsp.sd*1e4:.4f} cm²\n"
        f"#  le:   {tsp.le*1000:.4f} mH\n"
    )
