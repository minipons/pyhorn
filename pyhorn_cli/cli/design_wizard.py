"""Design wizard commands: resize-wizard, hornresp, chamber-wizard, segment-wizard."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml
from dataclasses import asdict

from pyhorn_core.config.parser import (
    parse_driver_specs,
    parse_horn_geometry,
    parse_horn_project,
)
from pyhorn_core.config.horn_models import HornGeometry
from pyhorn_core.config.project_models import HornProject
from pyhorn_core.solver.hornresp import solve_hornresp_profile
from pyhorn_core.solver.horn_segment import compute_horn_segment
from pyhorn_core.solver.chamber_wizard import (
    parse_tsp,
    compute_chamber_params,
    validate_chamber,
    build_yaml_snippet,
    ComputedChamberParams,
)
from pyhorn_core.solver.adapter import compute_throat_adapter

from ._shared import _horn_geometry_to_dict, _driver_specs_to_dict

from pyhorn_core.solver.synthesis_wizard import (
    SynthesisInput,
    synthesize_horn_system,
    synthesis_to_horn_geometry_yaml,
    synthesis_to_driver_yaml,
)


def resize_wizard(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Path to project YAML (geometry + metadata)"
    ),
    horn: Optional[Path] = typer.Option(
        None,
        "--horn",
        "-h",
        help="Path to geometry YAML (standalone, no project metadata)",
    ),
    driver: Path = typer.Option(
        ..., "--driver", "-d", help="Path to driver JSON/YAML config"
    ),
    factor: float = typer.Option(
        ...,
        "--factor",
        "-f",
        help="Linear resize factor (e.g. 1.5 = 50% larger; 0.8 = 20% smaller)",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for resized geometry YAML (defaults to <input stem>_resized.yaml)",
    ),
    adjust_sd: bool = typer.Option(
        True,
        "--adjust-sd/--no-adjust-sd",
        help="Scale driver piston area (Sd) by factor² (default: enabled)",
    ),
    adjust_re: bool = typer.Option(
        False,
        "--adjust-re/--no-adjust-re",
        help="Scale driver voice coil resistance (Re) by factor² (default: disabled; "
        "Re of the same driver does not change with size)",
    ),
    geometry_only: bool = typer.Option(
        False,
        "--geometry-only",
        help="Write only the resized geometry YAML (no project YAML with metadata)",
    ),
):
    """Resize Wizard — scale a horn geometry proportionally (Hornresp page 68).

    Applies proportional scaling to horn dimensions and optionally the driver Sd.
    The response curve shape is preserved; frequency axis shifts by 1/factor
    (larger horn → lower cutoff frequency).

    Scaling rules:
      Throat area (S1), mouth area (S2)   → × factor²
      Path length (L12)                   → × factor
      Driver Sd                          → × factor²  (disabled with --no-adjust-sd)
      Driver Re                          → × factor²  (disabled by default; rarely needed)
      Driver Mms, BL, CMS, RMS, VAS     → unchanged (same driver)

    Example: scale the HiroB horn up by 50% (larger → lower cutoff)
        pyhorn resize-wizard --project projects/hirob.yaml --driver drivers/fe166nv2.yaml \\
            --factor 1.5 --output resized_hirob.yaml

    Example: scale down by 20%
        pyhorn resize-wizard --project projects/hirob.yaml --driver drivers/fe166nv2.yaml \\
            --factor 0.8 --output smaller_hirob.yaml
    """
    from pyhorn_core.solver.resize import apply_resize

    horn_proj: Optional[HornProject] = None
    if project is None and horn is None:
        typer.secho("Error: specify either --project or --horn", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    assert horn is not None  # guaranteed by above check

    if factor <= 0:
        typer.secho(
            f"Error: resize factor must be positive, got {factor}", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # 1. Load driver
    try:
        driver_specs = parse_driver_specs(driver)
    except Exception as e:
        typer.secho(f"Error loading driver specs: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 2. Load geometry
    try:
        if project is not None:
            horn_proj, horn_geo = parse_horn_project(project)
            input_name = horn_proj.name or project.stem
            is_project = True
        else:
            horn_geo = parse_horn_geometry(horn)
            input_name = horn.stem
            is_project = False
    except Exception as e:
        typer.secho(f"Error loading geometry: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # 3. Apply resize
    resized_geo, resized_driver = apply_resize(
        horn_geo, driver_specs, factor, adjust_sd=adjust_sd, adjust_re=adjust_re
    )

    # 4. Determine output path
    if output is None:
        suffix = "_resized.yaml"
        if is_project:
            assert project is not None
            output = project.parent / f"{project.stem}{suffix}"
        else:
            assert horn is not None
            output = horn.parent / f"{horn.stem}{suffix}"

    # 5. Write geometry YAML
    geo_data = _horn_geometry_to_dict(resized_geo)
    geo_yaml_str = yaml.safe_dump(geo_data, sort_keys=False, default_flow_style=False)
    geo_path = (
        output.with_suffix(".yaml")
        if output.suffix not in (".yaml", ".yml")
        else output
    )
    geo_path.write_text(geo_yaml_str, encoding="utf-8")

    # 6. If project mode and not --geometry-only, write project YAML + driver YAML
    if is_project and not geometry_only:
        assert horn_proj is not None
        driver_path = output.parent / f"{output.stem}_driver.yaml"
        driver_data = _driver_specs_to_dict(resized_driver)
        driver_yaml_str = yaml.safe_dump(
            driver_data, sort_keys=False, default_flow_style=False
        )
        driver_path.write_text(driver_yaml_str, encoding="utf-8")

        # Build project YAML with resized references
        proj_data = {
            "name": f"{horn_proj.name} (resized ×{factor})" if horn_proj.name else None,
            "geometry_path": geo_path.name,
            "driver_coord": (
                [round(c, 4) for c in horn_proj.driver_coord]
                if horn_proj.driver_coord
                else None
            ),
            "width": round(horn_proj.width, 4) if horn_proj.width is not None else None,
            "enclosure": (
                [round(v, 4) for v in horn_proj.enclosure]
                if horn_proj.enclosure
                else None
            ),
            "thickness": round(horn_proj.thickness, 4) if horn_proj.thickness else None,
            "material": horn_proj.material,
            "notes": horn_proj.notes,
        }
        # Carry over rear_chamber, throat_chamber, vented_box, passive_radiator if present
        for key in ("rear_chamber", "throat_chamber", "vented_box", "passive_radiator"):
            val = getattr(horn_proj, key, None)
            if val is not None:
                proj_data[key] = asdict(val)  # type: ignore[literal-required]

        proj_yaml_str = yaml.safe_dump(
            {k: v for k, v in proj_data.items() if v is not None},
            sort_keys=False,
            default_flow_style=False,
        )
        proj_path = (
            output.with_suffix(".project.yaml") if output.suffix == ".yaml" else output
        )
        proj_path.write_text(proj_yaml_str, encoding="utf-8")

        typer.secho(
            f"✓ Resize Wizard complete  (factor ×{factor})", fg=typer.colors.GREEN
        )
        typer.echo(f"  Resized geometry: {geo_path}")
        typer.echo(f"  Resized driver:   {driver_path}")
        typer.echo(f"  Resized project:  {proj_path}")
    else:
        typer.secho(
            f"✓ Resize Wizard complete  (factor ×{factor})", fg=typer.colors.GREEN
        )
        typer.echo(f"  Resized geometry: {geo_path}")


def hornresp(
    s1: Optional[float] = typer.Option(None, help="Hornresp S1 throat area (cm²)"),
    s2: Optional[float] = typer.Option(None, help="Hornresp S2 mouth area (cm²)"),
    f12: Optional[float] = typer.Option(
        None, help="Hornresp F12 flare cutoff (Hz) [Exp/Hyp profiles]"
    ),
    t: Optional[float] = typer.Option(
        None, "--t", help="Hornresp hyperbolic T parameter [Hyp profile only]"
    ),
    hyp: Optional[float] = typer.Option(
        None,
        "--hyp",
        help="Hornresp Hyp / L12 horn axis length (cm) [Hyp/Con/Par/Cat profiles]",
    ),
    l12: Optional[float] = typer.Option(
        None,
        "--l12",
        help="Alias for --hyp; horn axis length in cm [Con/Exp/Par/Cat profiles]",
    ),
    profile_type: str = typer.Option(
        "hyp",
        "--profile-type",
        help="Horn profile type: hyp (hyperbolic) | exp (exponential) | con (conical) | par (parabolic) | cat (catenoidal)",
    ),
    output_yaml: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional output YAML path for the generated horn",
    ),
    enclosure_type: str = typer.Option(
        "BLH", "--enclosure", help="Enclosure type: FLH or BLH"
    ),
    n_segments: int = typer.Option(
        100, help="Continuous-profile discretisation segments"
    ),
    lrc: float = typer.Option(0.0, help="Rear chamber average length (m)"),
    vrc: Optional[float] = typer.Option(None, help="Rear chamber volume (m³)"),
    vtc: float = typer.Option(0.0, help="Throat chamber volume (m³)"),
):
    """Solve Hornresp S1/S2/F12 inputs for any profile type and emit a pyhorn horn definition.

    Profile types supported:
      hyp    — Hyperbolic (cosh): Area(x) = S1·(cosh(m·x) + T·sinh(m·x))²  [Hornresp 'Hyp']
      exp    — Exponential: Area(x) = S1·exp(m·x),  F12 = c·m/(4π)          [Hornresp 'Exp']
      con    — Conical (linear): Area(x) = S1 + (S2−S1)·(x/L12)            [Hornresp 'Con']
      par    — Parabolic (quadratic): Area(x) = S1·(1−(1−√(S2/S1))·x/L12)² [Hornresp 'Par']
      cat    — Catenoidal: Area(x) = S1·cosh²(m·x),  T=1                    [Hornresp 'Cat']

    Parameter rules per profile type:
      hyp:  Provide any 4 of S1, S2, F12, T, Hyp. The 5th is solved.
      exp:  Provide any 2 of S1, S2, F12, L12. The other two are solved.
      con:  Provide S1, S2, L12. F12 is N/A for conical.
      par:  Provide S1, S2, L12. F12 is N/A for parabolic.
      cat:  Provide S1, S2, L12. F12 is derived from catenoidal geometry.

    Examples
    --------
    Hyperbolic (Hornresp Hyp mode, T=0.3 given, Hyp solved):
        pyhorn hornresp --s1 40 --s2 300 --f12 50.43 --t 0.3 --hyp 152.7 \\
            --profile-type hyp

    Exponential (F12=50 Hz given, L12 solved):
        pyhorn hornresp --s1 40 --s2 300 --f12 50 --profile-type exp

    Exponential (L12=150 cm given, F12 derived):
        pyhorn hornresp --s1 40 --s2 300 --l12 150 --profile-type exp

    Conical (linear expansion):
        pyhorn hornresp --s1 40 --s2 300 --l12 150 --profile-type con

    Catenoidal (minimum-length optimal):
        pyhorn hornresp --s1 40 --s2 300 --l12 150 --profile-type cat
    """
    try:
        solved = solve_hornresp_profile(
            profile_type=profile_type,
            s1_cm2=s1,
            s2_cm2=s2,
            f12_hz=f12,
            t=t,
            hyp_cm=hyp,
            l12_cm=l12,
        )
    except Exception as e:
        typer.secho(f"Error solving Hornresp inputs: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    profile_type_str = str(solved["profile_type"])
    pt_label = {
        "conical": "Conical",
        "exp": "Exponential",
        "par": "Parabolic",
        "cat": "Catenoidal",
        "hyp": "Hyperbolic",
    }.get(profile_type_str, profile_type_str.title())

    typer.echo(f"Solved Hornresp {pt_label} horn parameters:")
    typer.echo(f"  Profile type: {pt_label}")
    typer.echo(f"  S1 (throat):  {solved['s1_cm2']:.3f} cm²")
    typer.echo(f"  S2 (mouth):   {solved['s2_cm2']:.3f} cm²")
    typer.echo(f"  L12 (length): {solved['l12_cm']:.3f} cm")
    f12_hz_val = float(solved["f12_hz"])
    if f12_hz_val > 0:
        typer.echo(f"  F12 (cutoff):  {f12_hz_val:.3f} Hz")
    else:
        typer.echo(f"  F12 (cutoff):  N/A ({pt_label} has no exponential cutoff)")
    if profile_type_str in ("hyperbolic", "catenoidal"):
        typer.echo(f"  T:             {solved['t']:.6f}")
    typer.echo(f"  Throat area:   {solved['throat_area_m2']:.6f} m²")
    typer.echo(f"  Mouth area:    {solved['mouth_area_m2']:.6f} m²")

    # Build YAML in sections format (Gap Analysis requirement)
    profile_type_out = solved["profile_type"]
    throat_area = float(solved["throat_area_m2"])
    mouth_area = float(solved["mouth_area_m2"])
    path_length = float(solved["path_length_m"])
    section: dict = {
        "name": "main_horn",
        "profile_type": profile_type_out,
        "start_area": round(throat_area, 8),
        "end_area": round(mouth_area, 8),
        "length": round(path_length, 6),
    }
    if profile_type_str in ("hyperbolic", "catenoidal"):
        section["hyperbolic_t"] = round(float(solved["t"]), 6)

    data: dict = {
        "enclosure_type": enclosure_type.upper(),
        "sections": [section],
        "n_segments": n_segments,
    }
    if lrc > 0 or vrc is not None:
        data["rear_chamber"] = {
            "lrc": lrc,
            "vrc": vrc if vrc is not None else lrc * throat_area,
        }
    if vtc > 0:
        data["throat_chamber"] = {"vtc": vtc}

    yaml_text = yaml.safe_dump(data, sort_keys=False, default_flow_style=False)
    if output_yaml is not None:
        solved_f12 = float(solved["f12_hz"])
        solved_t = float(solved.get("t", 0.0))
        header = (
            f"# Generated from Hornresp {pt_label} inputs\n"
            f"# S1={solved['s1_cm2']:.3f} cm², S2={solved['s2_cm2']:.3f} cm², "
            f"L12={solved['l12_cm']:.3f} cm"
        )
        if solved_f12 > 0:
            header += f", F12={solved_f12:.3f} Hz"
        if profile_type_out in ("hyperbolic", "catenoidal"):
            header += f", T={solved_t:.6f}"
        header += f"\n# Profile type: {profile_type_out}\n"
        output_yaml.write_text(header + yaml_text, encoding="utf-8")
        typer.echo(f"Saved {output_yaml}")
    else:
        typer.echo("")
        typer.echo(yaml_text.rstrip())


# ── Chamber Design Wizard ─────────────────────────────────────────────────────

_ASCII_SIGNAL_CHAIN = """
  ┌─────────────────────────────────────────────────────────────┐
  │              🔊 Driver  →  📦 Rear  →  🔶 Throat  →  ⚪ Aperture  →  🐚 Horn       │
  │              Cone        Chamber       Chamber          (Ap1/Lpt)        Throat      │
  │                           Vrc/Lrc       Vtc/Atc                                            │
  └─────────────────────────────────────────────────────────────┘
"""

_PARAM_HELP = {
    "vrc_L": (
        "Vrc (litres)",
        "Rear chamber volume — too small = peaky response, too large = no loading.\n"
        "                   Vrc = Vas × (Qts²/Qts_target² − 1). Target Qts ≈ 0.6.",
    ),
    "lrc_cm": (
        "Lrc (cm)",
        "Rear chamber path length — derived from Vrc & 200×200 mm default box section.",
    ),
    "vtc_cm3": (
        "Vtc (cm³)",
        "Throat chamber volume — dust cap clearance & acoustic coupling. Fixed ~100 cm³.",
    ),
    "atc_cm2": (
        "Atc (cm²)",
        "Throat chamber area — should be ≈ Sd for good acoustic coupling to driver.",
    ),
    "ap1_cm2": (
        "Ap1 (cm²)",
        "Baffle aperture area — controls coupling to the horn throat. ~50 mm dia hole.",
    ),
    "lpt_cm": (
        "Lpt (cm)",
        "Baffle / neck thickness — affects throat resonance. Fixed ~12 mm.",
    ),
}

_PARAM_RANGES = {
    "vrc_L": (0.5, 30.0),
    "lrc_cm": (3.0, 80.0),
    "vtc_cm3": (5.0, 500.0),
    "atc_cm2": (5.0, 100.0),
    "ap1_cm2": (1.0, 80.0),
    "lpt_cm": (3.0, 50.0),
}


def _prompt_param(
    name: str,
    current: Optional[float],
    unit: str,
    param_range: tuple[float, float],
    warning: Optional[str],
    help_text: str,
) -> float:
    """Prompt for a single numeric parameter, returning the (possibly updated) value."""
    typer.echo("")
    typer.secho(f"  ── {name} ──", fg=typer.colors.CYAN)
    typer.secho(f"    {help_text}", fg=typer.colors.WHITE)
    if warning:
        typer.secho(f"    ⚠  {warning}", fg=typer.colors.YELLOW)
    typer.echo(f"    Range: {param_range[0]}–{param_range[1]} {unit}")
    typer.echo(
        f"    Current: {current:.4g} {unit}"
        if current is not None
        else "    Current: (not set)"
    )
    raw = typer.prompt("    New value (Enter to keep current, 'q' to quit wizard)")
    if raw.strip().lower() == "q":
        raise typer.Exit(code=0)
    if not raw.strip():
        if current is None:
            typer.secho(
                "    No current value — using default range midpoint.",
                fg=typer.colors.YELLOW,
            )
            return (param_range[0] + param_range[1]) / 2.0
        return current
    try:
        val = float(raw.strip())
    except ValueError:
        typer.secho("    Invalid number — keeping current value.", fg=typer.colors.RED)
        return (
            current if current is not None else (param_range[0] + param_range[1]) / 2.0
        )
    if val < param_range[0] or val > param_range[1]:
        typer.secho(
            f"    Value {val} outside range [{param_range[0]}, {param_range[1]}] — clamping.",
            fg=typer.colors.YELLOW,
        )
        val = max(param_range[0], min(param_range[1], val))
    return val


def chamber_wizard(
    driver_yaml: Optional[Path] = typer.Option(
        None, "--driver", "-d", help="Path to driver YAML/JSON file (T-S parameters)"
    ),
    qts_target: float = typer.Option(
        0.6,
        "--qts-target",
        help="Target Qts for rear-chamber alignment (typical range 0.5–0.7). Default: 0.6.",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        help="Run in interactive mode (default). Use --no-interactive for batch.",
    ),
    box_width_mm: float = typer.Option(
        200.0,
        "--box-width",
        help="Box internal width in mm for Lrc derivation (default 200 mm → 0.04 m² section).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the YAML snippet. "
        "In interactive mode, defaults to <driver stem>_chamber.yaml. "
        "In batch mode, prints to stdout if omitted.",
    ),
):
    """Chamber Design Wizard — estimate Vrc, Lrc, Vtc, Atc, Ap1, Lpt from driver T-S parameters.

    This interactive wizard:
      1. Shows the acoustic signal chain (ASCII diagram)
      2. Loads T-S parameters from a driver YAML file
      3. Computes recommended chamber values (target Qts = 0.6)
      4. Displays each parameter with its physical meaning
      5. Lets you override any value with validation
      6. Outputs a YAML snippet ready to paste into your project file

    Example
    -------
        pyhorn chamber-wizard --driver drivers/FE166NV2.yaml

        # Non-interactive (just compute and print YAML):
        pyhorn chamber-wizard --driver drivers/FE166NV2.yaml --no-interactive
    """
    # ── 1. Load driver YAML ───────────────────────────────────────────────────
    if driver_yaml is None:
        typer.secho(
            "No --driver specified. Enter path to driver YAML (or 'q' to quit):",
            fg=typer.colors.YELLOW,
        )
        raw = typer.prompt("  Driver YAML path")
        if raw.strip().lower() == "q":
            raise typer.Exit(code=0)
        driver_yaml = Path(raw.strip())

    if not driver_yaml.exists():
        typer.secho(f"Driver file not found: {driver_yaml}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    try:
        yaml_text = driver_yaml.read_text(encoding="utf-8")
    except Exception as e:
        typer.secho(f"Error reading driver YAML: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # ── 2. Parse T-S params ───────────────────────────────────────────────────
    tsp = parse_tsp(yaml_text)

    if tsp.fs is None or tsp.qts is None or tsp.vas is None:
        typer.secho(
            "Could not parse all required T-S parameters (fs, qts, vas) from driver YAML.\n"
            "Make sure your driver YAML contains at least: fs, qts, vas.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # ── 3. Signal chain diagram ───────────────────────────────────────────────
    typer.echo(
        typer.style("\n🎛  Chamber Design Wizard", fg=typer.colors.CYAN, bold=True)
    )
    typer.echo(
        typer.style(
            "    Estimate Vrc, Lrc, Vtc, Atc, Ap1, Lpt from driver T-S parameters\n",
            fg=typer.colors.WHITE,
        )
    )
    typer.echo(_ASCII_SIGNAL_CHAIN)

    # ── 4. T-S parameter summary ─────────────────────────────────────────────
    typer.echo("Loaded driver T-S parameters:")
    typer.echo(f"  fs   = {tsp.fs:.1f} Hz")
    typer.echo(f"  Qts  = {tsp.qts:.3f}")
    typer.echo(f"  Vas  = {tsp.vas * 1e6:.1f} L  ({tsp.vas:.5f} m³)")
    if tsp.sd_cm2 is not None:
        typer.echo(f"  Sd   = {tsp.sd_cm2:.1f} cm²  ({tsp.sd:.5f} m²)")
    else:
        typer.echo(f"  Sd   = (not in YAML — using 58 mm driver diameter fallback)")

    # ── 5. Compute recommended values ────────────────────────────────────────
    import pyhorn_core.solver.chamber_wizard as _cw

    _cw.QTS_TARGET = qts_target

    p = compute_chamber_params(tsp)
    warnings = validate_chamber(p, tsp, qts_target)

    typer.echo(f"\n{'─'*62}")
    typer.secho(
        "  Recommended values (target Qts = {0:.2f}):".format(qts_target),
        fg=typer.colors.CYAN,
    )
    typer.echo(f"{'─'*62}")

    params_display = [
        ("Vrc", p.vrc_L, "L", _PARAM_HELP["vrc_L"][1], warnings.vrc_warning),
        ("Lrc", p.lrc_m * 100.0, "cm", _PARAM_HELP["lrc_cm"][1], warnings.lrc_warning),
        ("Vtc", p.vtc_cm3, "cm³", _PARAM_HELP["vtc_cm3"][1], warnings.vtc_warning),
        ("Atc", p.atc_cm2, "cm²", _PARAM_HELP["atc_cm2"][1], warnings.atc_warning),
        ("Ap1", p.ap1_cm2, "cm²", _PARAM_HELP["ap1_cm2"][1], warnings.ap1_warning),
        ("Lpt", p.lpt_cm, "cm", _PARAM_HELP["lpt_cm"][1], warnings.lpt_warning),
    ]

    for name, val, unit, help_text, warning in params_display:
        warn_str = (
            typer.style(f"  ⚠  {warning}", fg=typer.colors.YELLOW) if warning else ""
        )
        typer.echo(f"  {name:<5} = {val:>8.4g} {unit}   {warn_str}")

    typer.echo(f"{'─'*62}")
    typer.secho(
        "\n  ⚠  Estimates only — use simulation to validate.\n",
        fg=typer.colors.YELLOW,
    )

    # ── 6. Interactive override ──────────────────────────────────────────────
    if interactive:
        typer.secho(
            "  Interactive mode — press Enter to accept each value,",
            fg=typer.colors.WHITE,
        )
        typer.secho(
            "  type a new number to override, or 'q' to quit and output current.\n",
            fg=typer.colors.WHITE,
        )

        BOX_AREA_M2 = (box_width_mm / 1000.0) ** 2  # square box assumption

        # Mutable copy of params
        vrc_L = p.vrc_L
        lrc_cm = p.lrc_m * 100.0
        vtc_cm3 = p.vtc_cm3
        atc_cm2 = p.atc_cm2
        ap1_cm2 = p.ap1_cm2
        lpt_cm = p.lpt_cm

        # Vrc → Lrc linkage (changing Vrc updates Lrc)
        try:
            vrc_L = _prompt_param(
                "Vrc (rear chamber volume)",
                vrc_L,
                "L",
                _PARAM_RANGES["vrc_L"],
                warnings.vrc_warning,
                _PARAM_HELP["vrc_L"][1]
                + f"\n    Changing Vrc also updates Lrc (box section = {box_width_mm}×{box_width_mm} mm).",
            )
            # Recompute Lrc from new Vrc
            lrc_cm = (vrc_L / 1000.0) / BOX_AREA_M2 * 100.0
            warnings = validate_chamber(
                ComputedChamberParams(
                    vrc_L=vrc_L,
                    lrc_m=lrc_cm / 100.0,
                    vtc_m3=vtc_cm3 / 1e6,
                    vtc_cm3=vtc_cm3,
                    atc_m2=atc_cm2 / 1e4,
                    atc_cm2=atc_cm2,
                    ap1_m2=ap1_cm2 / 1e4,
                    ap1_cm2=ap1_cm2,
                    lpt_m=lpt_cm / 100.0,
                    lpt_cm=lpt_cm,
                ),
                tsp,
                qts_target,
            )
        except typer.Exit:
            raise

        # Lrc
        lrc_cm = _prompt_param(
            "Lrc (rear chamber path length)",
            lrc_cm,
            "cm",
            _PARAM_RANGES["lrc_cm"],
            warnings.lrc_warning,
            _PARAM_HELP["lrc_cm"][1],
        )

        # Vtc
        vtc_cm3 = _prompt_param(
            "Vtc (throat chamber volume)",
            vtc_cm3,
            "cm³",
            _PARAM_RANGES["vtc_cm3"],
            warnings.vtc_warning,
            _PARAM_HELP["vtc_cm3"][1],
        )

        # Atc
        atc_cm2 = _prompt_param(
            "Atc (throat chamber area)",
            atc_cm2,
            "cm²",
            _PARAM_RANGES["atc_cm2"],
            warnings.atc_warning,
            _PARAM_HELP["atc_cm2"][1],
        )

        # Ap1
        ap1_cm2 = _prompt_param(
            "Ap1 (baffle aperture area)",
            ap1_cm2,
            "cm²",
            _PARAM_RANGES["ap1_cm2"],
            warnings.ap1_warning,
            _PARAM_HELP["ap1_cm2"][1],
        )

        # Lpt
        lpt_cm = _prompt_param(
            "Lpt (baffle thickness)",
            lpt_cm,
            "cm",
            _PARAM_RANGES["lpt_cm"],
            warnings.lpt_warning,
            _PARAM_HELP["lpt_cm"][1],
        )

        # Rebuild ComputedChamberParams from final values
        p = ComputedChamberParams(
            vrc_L=vrc_L,
            lrc_m=lrc_cm / 100.0,
            vtc_m3=vtc_cm3 / 1e6,
            vtc_cm3=vtc_cm3,
            atc_m2=atc_cm2 / 1e4,
            atc_cm2=atc_cm2,
            ap1_m2=ap1_cm2 / 1e4,
            ap1_cm2=ap1_cm2,
            lpt_m=lpt_cm / 100.0,
            lpt_cm=lpt_cm,
        )
        warnings = validate_chamber(p, tsp, qts_target)

    # ── 7. Final validation pass ──────────────────────────────────────────────
    final_warnings = validate_chamber(p, tsp, qts_target)
    any_warnings = any(
        getattr(final_warnings, f"{k}_warning")
        for k in ["vrc", "lrc", "vtc", "atc", "ap1", "lpt"]
    )
    if any_warnings:
        typer.secho("\n  Validation warnings:", fg=typer.colors.YELLOW)
        for k in ["vrc", "lrc", "vtc", "atc", "ap1", "lpt"]:
            w = getattr(final_warnings, f"{k}_warning")
            if w:
                typer.secho(f"    {k.upper()}: {w}", fg=typer.colors.YELLOW)

    # ── 8. Output YAML snippet ─────────────────────────────────────────────────
    yaml_snippet = build_yaml_snippet(p)

    typer.echo("\n" + "─" * 62)
    typer.secho("  YAML Snippet (paste into your project YAML):", fg=typer.colors.GREEN)
    typer.echo("─" * 62)
    typer.secho(f"\n{yaml_snippet}\n", fg=typer.colors.WHITE)
    typer.echo("─" * 62)

    # ── 9. Save to file option ─────────────────────────────────────────────────
    _out_path: Optional[Path] = None
    if output is not None:
        _out_path = output
    elif interactive:
        save = typer.prompt("\n  Save YAML snippet to a file? (y/N)")
        if save.strip().lower() == "y":
            default_path = driver_yaml.parent / f"{driver_yaml.stem}_chamber.yaml"
            out_path_str = typer.prompt(f"  Output path", default=str(default_path))
            _out_path = Path(out_path_str.strip())

    if _out_path is not None:
        content = (
            f"# Chamber Design Wizard output\n"
            f"# Driver: {driver_yaml.name}\n"
            f"# Target Qts: {qts_target}\n\n"
            f"{yaml_snippet}\n"
        )
        _out_path.write_text(content, encoding="utf-8")
        typer.secho(f"  Saved to {_out_path}", fg=typer.colors.GREEN)

    typer.secho("\nDone.", fg=typer.colors.GREEN)


def segment_wizard(
    s1: Optional[float] = typer.Option(
        None,
        "--s1",
        help="Throat (neck) area in cm²  [provide any 3 of --s1, --s2, --l12, --f12]",
    ),
    s2: Optional[float] = typer.Option(
        None,
        "--s2",
        help="Mouth area in cm²  [provide any 3 of --s1, --s2, --l12, --f12]",
    ),
    l12: Optional[float] = typer.Option(
        None,
        "--l12",
        help="Horn axis length in cm  [provide any 3 of --s1, --s2, --l12, --f12]",
    ),
    f12: Optional[float] = typer.Option(
        None,
        "--f12",
        help="Low-frequency cutoff in Hz  [provide any 3 of --s1, --s2, --l12, --f12]",
    ),
):
    """Horn Segment Wizard — geometry calculator for a single catenoidal horn segment.

    Given any 3 of (S1 throat area, S2 mouth area, L12 horn length, F12 cutoff freq),
    computes the 4th using the catenoidal (T=1) horn formulas.  Also prints the
    20-point catenoidal area profile and estimates the system volume (horn internal
    volume + estimated 0.1 L throat chamber).

    Catenoidal (T=1) formulas (c = 343 m/s):
      F12 = c/(2π) × √(S2/S1 − 1) / L12
      L12 = c/(2π × F12) × √(S2/S1 − 1)
      S2  = S1 / (1 + (2π×F12×L12/c)²)
      S1  = S2 / (1 + (2π×F12×L12/c)²)

    Example
    -------
    Compute cutoff frequency for a 40 cm²→300 cm² horn 150 cm long:

        pyhorn segment-wizard --s1 40 --s2 300 --l12 150

    Compute the required mouth area for a 40 cm² throat, 150 cm horn at 50 Hz cutoff:

        pyhorn segment-wizard --s1 40 --l12 150 --f12 50

    Compute horn length for a 40→300 cm² horn targeting 50 Hz cutoff:

        pyhorn segment-wizard --s1 40 --s2 300 --f12 50
    """
    # Convert cm² → m² for s1, s2; cm → m for l12
    s1_m2 = (s1 * 1e-4) if s1 is not None else None
    s2_m2 = (s2 * 1e-4) if s2 is not None else None
    l12_m = (l12 * 0.01) if l12 is not None else None

    try:
        result = compute_horn_segment(
            s1_m2=s1_m2,
            s2_m2=s2_m2,
            l12_m=l12_m,
            f12_hz=f12,
        )
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # ── Pretty-print the computed result ─────────────────────────────────────
    typer.echo(f"\n{'='*60}")
    typer.echo(f"  Horn Segment Wizard — catenoidal (T=1)")
    typer.echo(f"{'='*60}")

    param_labels = {
        "f12_hz": "Cutoff frequency F12",
        "l12_cm": "Horn length L12",
        "s2_cm2": "Mouth area S2",
        "s1_cm2": "Throat area S1",
    }
    label = param_labels.get(result.computed_param, result.computed_param)
    if result.computed_param == "f12_hz":
        typer.echo(f"\n  Computed {label}: {result.computed_value:.2f} Hz")
    elif result.computed_param == "l12_cm":
        typer.echo(
            f"\n  Computed {label}: {result.computed_value:.2f} cm  ({result.computed_value/100:.4f} m)"
        )
    else:
        typer.echo(f"\n  Computed {label}: {result.computed_value:.2f} cm²")

    # ── Area profile table ────────────────────────────────────────────────────
    typer.echo(f"\n  {'─'*50}")
    typer.echo(f"  {'Position':>12}  {'Area (cm²)':>12}  {'Diameter (mm)':>14}")
    typer.echo(f"  {'─'*50}")
    for frac, area_cm2 in result.area_profile:
        equiv_diameter_mm = 2.0 * math.sqrt(area_cm2 / math.pi) * 10.0
        if frac == 0.0:
            label = "throat"
        elif frac == 1.0:
            label = "mouth"
        else:
            label = ""
        pos_str = f"{frac:.2f}" if not label else f"{frac:.2f} ({label})"
        typer.echo(f"  {pos_str:>26}  {area_cm2:>12.2f}  {equiv_diameter_mm:>14.1f}")
    typer.echo(f"  {'─'*50}")

    # ── System volume ─────────────────────────────────────────────────────────
    typer.echo(
        f"\n  System volume: {result.system_volume_l:.3f} L  (horn + 0.1 L throat chamber)"
    )

    # ── Unit summary ─────────────────────────────────────────────────────────
    typer.echo(f"\n  Input parameters:")
    for param, val, unit in [
        ("S1 (throat)", s1, "cm²"),
        ("S2 (mouth)", s2, "cm²"),
        ("L12 (length)", l12, "cm"),
        ("F12 (cutoff)", f12, "Hz"),
    ]:
        if val is not None:
            typer.echo(f"    {param}: {val} {unit}")

    typer.echo(f"\n{'='*60}\n")


def synthesis_wizard(
    driver: Path = typer.Option(
        ..., "--driver", "-d", help="Path to driver YAML/JSON file with T-S parameters"
    ),
    f3: float = typer.Option(
        50.0,
        "--f3",
        help="Target -3 dB low-frequency cutoff in Hz (default: 50 Hz). "
        "Used to derive the horn cutoff F12 ≈ f3/1.2.",
    ),
    f7: float = typer.Option(
        8000.0,
        "--f7",
        help="Target -3 dB high-frequency cutoff in Hz (default: 8000 Hz).",
    ),
    qts_alignment: float = typer.Option(
        0.55,
        "--qts-alignment",
        help="Target Qts for the rear-chamber alignment (default: 0.55). "
        "Higher values = stronger loading, lower f3 but more peaked response.",
    ),
    max_mouth_area: float = typer.Option(
        0.12,
        "--max-mouth-area",
        help="Maximum horn mouth area in m² (default: 0.12 m² = 2400×1200 mm MDF budget).",
    ),
    max_path_length: float = typer.Option(
        2.0,
        "--max-path-length",
        help="Maximum horn path length in metres (default: 2.0 m for 2400 mm MDF sheet).",
    ),
    max_iter: int = typer.Option(
        50,
        "--max-iter",
        help="Max iterations for the internal horn optimiser (default: 50). "
        "Note: synthesis-wizard is primarily formula-based; this controls any "
        "iterative refinement step. Use --max-iter=5 for faster test runs.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to write the synthesised geometry YAML. "
        "Prints to stdout if omitted.",
    ),
):
    """Horn System Synthesis Wizard — synthesise a complete horn system from T-S parameters.

    Implements the Hornresp page 067 "System Design / Synthesis Wizard" algorithm:\n
      F12 = f3 / 1.2          (horn cutoff from target -3 dB cutoff)\n
      S1  = Sd                 (throat area ≈ driver piston area)\n
      S2  = constrained by MDF sheet (max_mouth_area × max_path_length)\n
      Vrc = Vas × (Qts² / Qts_align² − 1)   (rear chamber volume)\n
      Lrc = Vrc / Atc          (rear chamber path length)\n
      Vtc = 0.002 × Vas        (throat chamber volume)\n
      Atc = Sd                 (throat chamber area)

    The command outputs:\n
      1. A YAML geometry snippet (sections, rear_chamber, throat_chamber, throat_adapter)\n
      2. T-S parameter reference comments

    Example
    -------
        pyhorn synthesis-wizard --driver drivers/FE166NV2.yaml --f3 45 --output synth.yaml

        # Synthesise targeting 40 Hz f3 with 0.65 alignment Q:
        pyhorn synthesis-wizard -d drivers/FE166NV2.yaml --f3 40 --qts-alignment 0.65
    """
    import re as _re

    # ── 1. Load and parse driver YAML ───────────────────────────────────────
    if not driver.exists():
        typer.secho(f"Driver file not found: {driver}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    yaml_text = driver.read_text(encoding="utf-8")

    def _get(key: str) -> Optional[float]:
        """Extract a float from YAML by key (case-insensitive, multiline)."""
        pat = _re.compile(
            rf"^\s*{_re.escape(key)}\s*:\s*([\d.e+\-]+)", _re.MULTILINE | _re.IGNORECASE
        )
        m = pat.search(yaml_text)
        return float(m.group(1)) if m else None

    fs = _get("fs")
    qts = _get("qts")
    vas = _get("vas")
    sd = _get("sd")
    re_ = _get("re")
    bl = _get("bl")
    mms = _get("mms")
    cms = _get("cms")
    rms = _get("rms")
    qes = _get("qes")
    qms = _get("qms")
    le = _get("le") or 0.0

    missing = [
        k for k, v in [("fs", fs), ("qts", qts), ("vas", vas), ("sd", sd)] if v is None
    ]
    if missing:
        typer.secho(
            f"Missing required T-S parameters in driver YAML: {', '.join(missing)}. "
            "Add them to your driver YAML and try again.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    assert fs is not None and qts is not None and vas is not None and sd is not None

    # sd may be in cm² — check for sd_cm2 if sd is missing
    if sd is None:
        sd = _get("sd_cm2")
        if sd is not None:
            sd = sd / 1e4  # cm² → m²

    if sd is None:
        typer.secho(
            "Could not find 'sd' (piston area, m²) or 'sd_cm2' in driver YAML. "
            "Add sd: <value> (m²) or sd_cm2: <value> to your driver YAML.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # vas may be in litres — check for vas_l
    vas_m3 = vas
    if vas is not None:
        vas_l = _get("vas_l")
        if vas_l is not None:
            vas_m3 = vas_l / 1000.0  # L → m³

    # ── 2. Build SynthesisInput ──────────────────────────────────────────────
    inp = SynthesisInput(
        fs=fs,
        qts=qts,
        vas=vas_m3,
        sd=sd,
        re=re_ or 7.8,
        bl=bl or 7.79,
        mms=mms or 0.010,
        cms=cms or 1e-3,
        rms=rms or 1.0,
        qes=qes or 0.30,
        qms=qms or 2.80,
        le=le,
        f3_target_hz=f3,
        f7_target_hz=f7,
        system_type="BLH",
        enclosure_type="BLH",
        qts_alignment=qts_alignment,
        max_mouth_area_m2=max_mouth_area,
        max_path_length_m=max_path_length,
        baffle_thickness_m=0.012,
        room_gain_db=0.0,
    )

    # ── 3. Run synthesis ─────────────────────────────────────────────────────
    try:
        syn = synthesize_horn_system(inp)
    except Exception as e:
        typer.secho(f"Synthesis failed: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # ── 4. Print summary banner ──────────────────────────────────────────────
    typer.secho("\n  🎺 Horn System Synthesis Wizard", fg=typer.colors.CYAN, bold=True)
    typer.echo(
        f"  Driver: {driver.name}  |  Target f3: {f3:.0f} Hz  |  f7: {f7:.0f} Hz\n"
    )

    typer.echo(f"  {'─'*56}")
    typer.echo(f"  {'Horn geometry':<30} {'Value':>24}")
    typer.echo(f"  {'─'*56}")
    typer.echo(f"  {'Throat area (S1)':<30} {syn.horn.throat_area_m2*1e4:>20.2f} cm²")
    typer.echo(f"  {'Mouth area (S2)':<30} {syn.horn.mouth_area_m2*1e4:>20.2f} cm²")
    typer.echo(f"  {'Path length (L12)':<30} {syn.horn.path_length_m*100:>20.1f} cm")
    typer.echo(f"  {'Cutoff frequency (F12)':<30} {syn.horn.f12_computed_hz:>20.1f} Hz")
    typer.echo(f"  {'─'*56}")
    typer.echo(f"  {'Chambers':<30} {'Value':>24}")
    typer.echo(f"  {'─'*56}")
    typer.echo(f"  {'Rear chamber volume (Vrc)':<30} {syn.chambers.vrc_L:>20.2f} L")
    typer.echo(
        f"  {'Rear chamber length (Lrc)':<30} {syn.chambers.lrc_m*100:>20.1f} cm"
    )
    typer.echo(
        f"  {'Throat chamber volume (Vtc)':<30} {syn.chambers.vtc_m3*1e6:>20.1f} cm³"
    )
    typer.echo(
        f"  {'Throat chamber area (Atc)':<30} {syn.chambers.atc_m2*1e4:>20.2f} cm²"
    )
    typer.echo(f"  {'Baffle aperture (Ap1)':<30} {syn.chambers.ap1_m2*1e4:>20.2f} cm²")
    typer.echo(f"  {'Baffle thickness (Lpt)':<30} {syn.chambers.lpt_m*1000:>20.1f} mm")
    typer.echo(f"  {'─'*56}")

    if syn.warnings:
        typer.secho(f"  {'Warnings':<30}", fg=typer.colors.YELLOW)
        for w in syn.warnings:
            icon = {"INFO": "ℹ", "WARN": "⚠", "ERROR": "✖"}.get(w.severity, "•")
            fg = (
                typer.colors.YELLOW
                if w.severity == "WARN"
                else typer.colors.CYAN if w.severity == "INFO" else typer.colors.RED
            )
            typer.echo(f"  {icon} {w.field.upper()}: {w.message}")
        typer.echo(f"  {'─'*56}")

    # ── 5. Build YAML output ─────────────────────────────────────────────────
    geo_yaml = synthesis_to_horn_geometry_yaml(syn)
    drv_yaml = synthesis_to_driver_yaml(syn, inp)

    full_output = (
        f"# Horn System Synthesis Wizard output\n"
        f"# Command: synthesis-wizard --driver {driver} --f3 {f3} --f7 {f7} "
        f"--qts-alignment {qts_alignment}\n"
        f"# Target: f3={f3:.0f} Hz, f7={f7:.0f} Hz, Qts_align={qts_alignment}\n"
        f"# ── Driver reference (from {driver.name}) ──────────────────────────\n"
        f"{drv_yaml}\n"
        f"# ── Synthesised geometry ─────────────────────────────────────────────\n"
        f"{geo_yaml}\n"
    )

    # ── 6. Output ────────────────────────────────────────────────────────────
    if output is not None:
        output.write_text(full_output, encoding="utf-8")
        typer.secho(f"\n✓ Synthesis complete", fg=typer.colors.GREEN)
        typer.echo(f"  Saved to: {output}")
    else:
        typer.echo("\n" + full_output)
        typer.secho(
            "✓ Synthesis complete  (use --output to save to file)",
            fg=typer.colors.GREEN,
        )
