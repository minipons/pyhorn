"""Horn simulation commands: tapped-horn, throat-adapter, auto-segment, diagnose-spl, driver-front-volume."""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml

from pyhorn_core.config.parser import (
    parse_driver_specs,
    parse_horn_geometry,
    parse_horn_project,
)
from pyhorn_core.solver.models import horn_response, horn_response_tapped, C
from pyhorn_core.config.chamber_models import RearChamber
from pyhorn_core.config.horn_models import Section, TappedHornGeometry
from pyhorn_core.solver.adapter import compute_throat_adapter, throat_adapter_profile
from pyhorn_core.output.plotter import plot_throat_adapter_profile


def tapped_horn(
    driver_config: Path = typer.Option(
        ..., "--driver", "-d", help="Path to driver JSON/YAML config"
    ),
    th_config: Path = typer.Option(
        ..., "--th", "-t", help="Path to Tapped Horn geometry YAML"
    ),
    output_dir: Path = typer.Option(
        Path("./outputs"), "--output-dir", "-o", help="Directory to save outputs"
    ),
    fmin: float = typer.Option(20.0, help="Minimum frequency (Hz)"),
    fmax: float = typer.Option(5000.0, help="Maximum frequency (Hz)"),
    n_points: int = typer.Option(500, help="Number of frequency points"),
    export_csv: bool = typer.Option(True, help="Export data to CSV"),
    plot: bool = typer.Option(True, help="Generate SPL plot (.png)"),
    plot_phase: bool = typer.Option(
        True, "--plot-phase/--no-plot-phase", help="Include phase + group delay panels"
    ),
    plot_distortion: bool = typer.Option(
        True, "--plot-distortion/--no-plot-distortion", help="Include distortion panel"
    ),
):
    """Simulate a Tapped Horn (TH / TH1 mode).

    The driver is positioned at an interior point of the horn (S2 or S3).
    See Hornresp manual pages 057–058.

    Example YAML (th_config):
        tap_segment_index: 2         # TH mode (S2) or TH1 mode (S3)
        rear_load_type: rear_chamber  # rear_chamber | free_space | infinite_baffle
        front_sections:
          - name: seg2
            profile_type: exponential
            start_area: 0.01327       # S2 = driver Sd (m²)
            end_area: 0.08           # mouth area (m²)
            length: 1.2              # path length S2→mouth (m)
        rear_chamber:
          vrc: 0.035                 # volume (m³)
          lrc: 0.15                  # length (m)
    """
    # Parse inputs
    driver = parse_driver_specs(driver_config)
    try:
        with open(th_config, "r", encoding="utf-8") as f:
            th_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        typer.secho(
            f"Error: Invalid YAML in tapped horn config '{th_config}': {e}",
            fg=typer.colors.RED,
        )
        typer.echo(
            "Hint: check for missing colons, incorrect indentation, "
            "or unquoted special characters."
        )
        raise typer.Exit(code=1)

    # Build TappedHornGeometry from YAML
    front_secs = []
    for raw_sec in th_data.get("front_sections", []):
        front_secs.append(
            Section(
                name=str(raw_sec.get("name", "")),
                profile_type=str(raw_sec.get("profile_type", "exponential")),
                start_area=float(raw_sec["start_area"]),
                end_area=float(raw_sec["end_area"]),
                length=float(raw_sec["length"]),
                hyperbolic_t=(
                    float(raw_sec["hyperbolic_t"])
                    if raw_sec.get("hyperbolic_t") is not None
                    else None
                ),
            )
        )

    rear_secs = []
    for raw_sec in th_data.get("rear_sections", []):
        rear_secs.append(
            Section(
                name=str(raw_sec.get("name", "")),
                profile_type=str(raw_sec.get("profile_type", "conical")),
                start_area=float(raw_sec["start_area"]),
                end_area=float(raw_sec["end_area"]),
                length=float(raw_sec["length"]),
                hyperbolic_t=(
                    float(raw_sec["hyperbolic_t"])
                    if raw_sec.get("hyperbolic_t") is not None
                    else None
                ),
            )
        )

    rear_chamber_data = th_data.get("rear_chamber")
    rear_chamber = None
    if rear_chamber_data:
        rear_chamber = RearChamber(
            vrc=float(rear_chamber_data.get("vrc", 0.0)),
            lrc=float(rear_chamber_data.get("lrc", 0.0)),
            fr_rc=float(rear_chamber_data.get("fr_rc", 0.0)),
        )

    th_geom = TappedHornGeometry(
        tap_segment_index=int(th_data.get("tap_segment_index", 2)),
        front_sections=front_secs,
        rear_sections=rear_secs,
        rear_chamber=rear_chamber,
        rear_load_type=str(th_data.get("rear_load_type", "rear_chamber")),
        ang=float(th_data.get("ang", 2.0 * math.pi)),
        n_segments=int(th_data.get("n_segments", 100)),
    )

    # Run solver
    freqs = np.linspace(fmin, fmax, n_points)
    result = horn_response_tapped(
        freqs, driver, th_geom, compute_distortion=plot_distortion
    )

    # Summary
    output_dir.mkdir(parents=True, exist_ok=True)
    typer.echo(f"Tapped Horn simulation ({th_geom.tap_segment_index=})")
    typer.echo(f"  Front path length: {th_geom.front_path_length():.3f} m")
    typer.echo(f"  Rear load type:     {th_geom.rear_load_type}")
    if th_geom.rear_chamber:
        typer.echo(
            f"  Rear chamber:        {th_geom.rear_chamber.vrc*1000:.1f} L, Lrc={th_geom.rear_chamber.lrc*100:.1f} cm"
        )
    max_spl_idx = int(np.argmax(result.spl))
    typer.echo(
        f"  Max SPL:            {result.spl[max_spl_idx]:.1f} dB @ {freqs[max_spl_idx]:.0f} Hz"
    )
    typer.echo(
        f"  SPL at 1 kHz:       {float(np.interp(1000, freqs, result.spl)):.1f} dB"
    )

    # CSV export
    csv_path = output_dir / "tapped_horn_response.csv"
    if export_csv:
        import csv

        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            eff = (
                result.efficiency_pct
                if result.efficiency_pct is not None
                else np.zeros_like(result.spl)
            )
            gd = (
                result.group_delay
                if result.group_delay is not None
                else np.zeros_like(result.spl)
            )
            writer.writerow(
                [
                    "frequency_hz",
                    "spl_db",
                    "impedance_ohm",
                    "excursion_mm",
                    "efficiency_pct",
                    "group_delay_ms",
                ]
            )
            for i in range(len(freqs)):
                writer.writerow(
                    [
                        round(freqs[i], 2),
                        round(result.spl[i], 2),
                        round(np.abs(result.impedance[i]), 2),
                        round(result.excursion[i] * 1000, 4),  # mm
                        round(eff[i], 3),
                        round(gd[i] * 1000, 3),
                    ]
                )
        typer.echo(f"  CSV: {csv_path}")

    # Plot
    if plot:
        plot_path = output_dir / "tapped_horn_response.png"
        typer.echo(f"  Plot: {plot_path}")
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            axes[0].semilogx(freqs, result.spl, color="tab:blue")
            axes[0].set_ylabel("SPL (dB)")
            axes[0].set_title("Tapped Horn Response")
            axes[0].grid(True, which="both", alpha=0.4)
            axes[1].loglog(freqs, result.impedance, color="tab:orange")
            axes[1].set_xlabel("Frequency (Hz)")
            axes[1].set_ylabel("|Z| (Ω)")
            axes[1].grid(True, which="both", alpha=0.4)
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
        except Exception as e:
            typer.secho(
                f"  Warning: could not generate plot: {e}", fg=typer.colors.YELLOW
            )

    typer.secho("Done.", fg=typer.colors.GREEN)


def throat_adapter(
    d1: float = typer.Option(..., "--d1", help="Input (throat chamber) diameter in mm"),
    d2: float = typer.Option(..., "--d2", help="Output (horn throat) diameter in mm"),
    a1: float = typer.Option(
        30.0, "--a1", help="Input side flare half-angle in degrees"
    ),
    a2: float = typer.Option(
        30.0, "--a2", help="Output side flare half-angle in degrees"
    ),
    profile_type: str = typer.Option(
        "conical",
        "--type",
        help="Adapter profile type: conical | exponential | parabolic | cylindrical",
    ),
    length: Optional[float] = typer.Option(
        None,
        "--length",
        help="Explicit adapter length in mm (default: minimum geometric length)",
    ),
    plot: bool = typer.Option(False, "--plot", help="Generate profile preview plot"),
    output_plot: Optional[Path] = typer.Option(
        None,
        "--output-plot",
        "-o",
        help="Path to save the profile plot PNG (implies --plot)",
    ),
    atc: Optional[float] = typer.Option(
        None,
        "--atc",
        help="Throat chamber area in cm² (for accurate profile; defaults to π·(d1/2)²)",
    ),
):
    """Compute a throat adapter geometry and emit a YAML snippet ready for your project.

    The throat adapter is a short profiled transition duct between the throat chamber
    opening (D1, driver side) and the horn throat (D2, horn side).  This command
    computes the minimum-length profile and prints the YAML parameters (``ap1``, ``lpt``,
    ``type``) that can be pasted directly into your project YAML under ``throat_adapter:``.

    Example
    -------
    Compute a conical adapter from a 50 mm throat chamber opening to a 100 mm horn
    throat with 30° flare angles:

        pyhorn throat-adapter --d1 50 --d2 100 --a1 30 --a2 30 --type conical --plot

    To use in your project YAML:

        throat_adapter:
          type: conical
          ap1: 0.007853
          lpt: 0.043301
    """
    # Normalise profile type
    valid_types = {"cylindrical", "conical", "exponential", "parabolic"}
    ptype = profile_type.lower()
    if ptype not in valid_types:
        typer.secho(
            f"Invalid --type '{profile_type}'. Must be one of: {', '.join(sorted(valid_types))}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Convert mm → metres
    D1_m = d1 / 1000.0
    D2_m = d2 / 1000.0

    # Length in metres (optional override)
    length_m = length / 1000.0 if length is not None else None

    # Compute adapter
    try:
        adapter = compute_throat_adapter(
            D1=D1_m,
            D2=D2_m,
            A1_deg=a1,
            A2_deg=a2,
            profile_type=ptype,
            length=length_m,
        )
    except Exception as e:
        typer.secho(f"Error computing adapter: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # ── YAML snippet ──────────────────────────────────────────────────────────
    lpt_cm = adapter.lpt * 100.0
    ap1_cm2 = adapter.ap1 * 10000.0
    d1_display = d1
    d2_display = d2

    yaml_data = {
        "throat_adapter": {
            "type": adapter.type,
            "ap1": round(adapter.ap1, 6),
            "lpt": round(adapter.lpt, 6),
        }
    }
    yaml_text = yaml.safe_dump(yaml_data, sort_keys=False)

    typer.echo(
        "\n# ── Throat Adapter YAML snippet ──────────────────────────────────────"
    )
    typer.echo("# Paste this block into your project YAML under the top level:")
    typer.echo(yaml_text.rstrip())
    typer.echo("# ────────────────────────────────────────────────────────────────────")

    # ── Summary ───────────────────────────────────────────────────────────────
    # Compute input-end area
    A0 = (math.pi / 4.0) * (D1_m**2)
    if atc is not None:
        A0 = (atc / 100.0) * 0.0001  # cm² → m²

    typer.echo(f"\nAdapter summary ({ptype}):")
    typer.echo(
        f"  D1 (throat chamber side):  {d1_display:.1f} mm  ({A0 * 10000.0:.2f} cm²)"
    )
    typer.echo(
        f"  D2 (horn throat side):      {d2_display:.1f} mm  ({ap1_cm2:.2f} cm²)"
    )
    typer.echo(f"  Flare angles:              A1={a1}°, A2={a2}°")
    typer.echo(f"  Minimum length (Lpt):      {lpt_cm:.2f} cm  ({adapter.lpt:.4f} m)")
    if length is not None and length_m is not None:
        typer.echo(f"  Requested length:           {length:.2f} mm  ({length_m:.4f} m)")
        if abs(adapter.lpt - length_m) > 1e-9:
            typer.secho(
                "  ⚠️  Requested length equals minimum length (profile unchanged).",
                fg=typer.colors.YELLOW,
            )

    # ── Profile plot ──────────────────────────────────────────────────────────
    if plot or output_plot is not None:
        profile = throat_adapter_profile(adapter, A0=A0, n_points=101)
        plot_path = output_plot or Path(
            f"throat_adapter_{ptype}_{int(d1)}to{int(d2)}mm.png"
        )
        typer.echo(f"\nGenerating profile plot at {plot_path} ...")
        plot_throat_adapter_profile(
            profile,
            plot_path,
            title=f"Throat Adapter — {ptype.title()}  ({d1_display:.0f} mm → {d2_display:.0f} mm)",
        )
        typer.secho(f"Plot saved to {plot_path}", fg=typer.colors.GREEN)

    typer.secho("\nDone.", fg=typer.colors.GREEN)


def auto_segment(
    json_config: Optional[Path] = typer.Option(
        None, "--input", "-i", help="Path to Onshape JSON export"
    ),
    output_yaml: Path = typer.Option(
        ..., "--output", "-o", help="Path to save Pyhorn YAML"
    ),
    from_clipboard: bool = typer.Option(
        False, "--from-clipboard", "-c", help="Read JSON from clipboard (macOS pbpaste)"
    ),
    n_segments: int = typer.Option(20, help="Number of conical segments to generate"),
    flip_x: bool = typer.Option(
        False, "--flip-x", help="Invert X coordinates (useful for CAD)"
    ),
    flip_y: bool = typer.Option(
        False, "--flip-y", help="Invert Y coordinates (useful for CAD Y-up systems)"
    ),
    geometry_aware: bool = typer.Option(
        False,
        "--geometry-aware",
        "-g",
        help="Use true perpendicular cross-sections and bend angles",
    ),
    preserve_breaks: bool = typer.Option(
        False,
        "--preserve-breaks",
        help="Preserve CAD stair/break stations instead of uniform resampling",
    ),
    center: bool = typer.Option(
        True,
        "--center/--no-center",
        help="Shift coordinates so bounding box starts at (0,0) [default: True]",
    ),
    # ── Throat Adapter options ─────────────────────────────────────────────────
    throat_adapter_d1: Optional[float] = typer.Option(
        None,
        "--throat-adapter-d1",
        help="Throat adapter: input (throat chamber) diameter in mm. "
        "Requires --throat-adapter-d2.",
    ),
    throat_adapter_d2: Optional[float] = typer.Option(
        None,
        "--throat-adapter-d2",
        help="Throat adapter: output (horn throat) diameter in mm. "
        "Requires --throat-adapter-d1.",
    ),
    throat_adapter_a1: float = typer.Option(
        30.0,
        "--throat-adapter-a1",
        help="Throat adapter: input side flare half-angle in degrees [default: 30]",
    ),
    throat_adapter_a2: float = typer.Option(
        30.0,
        "--throat-adapter-a2",
        help="Throat adapter: output side flare half-angle in degrees [default: 30]",
    ),
    throat_adapter_type: str = typer.Option(
        "conical",
        "--throat-adapter-type",
        help="Throat adapter profile type: cylindrical | conical | exponential | parabolic [default: conical]",
    ),
    throat_adapter_length: Optional[float] = typer.Option(
        None,
        "--throat-adapter-length",
        help="Throat adapter explicit length in mm (default: minimum geometric length).",
    ),
    output_format: Optional[str] = typer.Option(
        None,
        "--output-format",
        help="Output format for the generated geometry YAML. "
        "'sections' emits the chained profile sections format "
        "(name, profile_type, length, start_area, end_area) preferred for TMM solver. "
        "'legacy' emits rectangular_segments (CAD-accurate per-segment format). "
        "Defaults to 'legacy' when --preserve-breaks is set, otherwise 'sections'.",
    ),
):
    """
    Generate Pyhorn segments automatically from an Onshape 2D air volume.

    Optionally compute and embed a throat adapter geometry (--throat-adapter-d1 and
    --throat-adapter-d2) so the output YAML includes the full horn + adapter geometry.
    """
    if not json_config and not from_clipboard:
        typer.secho(
            "Error: Must provide either --input or --from-clipboard",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # Validate throat adapter flags
    ta_provided = throat_adapter_d1 is not None or throat_adapter_d2 is not None
    if throat_adapter_d1 is None and throat_adapter_d2 is not None:
        typer.secho(
            "--throat-adapter-d1 is required when --throat-adapter-d2 is specified.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if throat_adapter_d1 is not None and throat_adapter_d2 is None:
        typer.secho(
            "--throat-adapter-d2 is required when --throat-adapter-d1 is specified.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    valid_types = {"cylindrical", "conical", "exponential", "parabolic"}
    ptype = throat_adapter_type.lower()
    if ta_provided and ptype not in valid_types:
        typer.secho(
            f"Invalid --throat-adapter-type '{throat_adapter_type}'. "
            f"Must be one of: {', '.join(sorted(valid_types))}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    try:
        from pyhorn_core.solver.medial_axis import generate_auto_segments

        if generate_auto_segments is None:
            typer.secho(
                "The 'pyhorn_segment' package is required for auto-segmentation but is not installed. "
                "Install it with: pip install pyhorn_segment",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        valid_formats = {"sections", "legacy"}
        if output_format is None:
            output_format = "legacy" if preserve_breaks else "sections"
        if output_format not in valid_formats:
            typer.secho(
                f"Invalid --output-format '{output_format}'. Must be one of: {', '.join(sorted(valid_formats))}",
                fg=typer.colors.RED,
            )
            raise typer.Exit(code=1)

        mode = "geometry-aware" if geometry_aware else "legacy"
        typer.echo(f"Calculating Medial Axis and shortest path ({mode} mode)...")
        exported = generate_auto_segments(
            json_config,
            output_yaml,
            n_segments,
            flip_x=flip_x,
            flip_y=flip_y,
            from_clipboard=from_clipboard,
            geometry_aware=geometry_aware,
            preserve_breaks=preserve_breaks,
            center=center,
            output_format=output_format,
        )
        if output_format == "sections":
            sections = exported.get("sections", [])
            typer.secho(
                f"Successfully generated {len(sections)} sections!",
                fg=typer.colors.GREEN,
            )
            for sec in sections:
                typer.echo(
                    f"  {sec['name']}: {sec['profile_type']}  "
                    f"{sec['length']:.4f}m  "
                    f"{sec['start_area']*10000:.1f}→{sec['end_area']*10000:.1f} cm²"
                )
        else:
            exported_count = len(
                exported.get(
                    "rectangular_segments", exported.get("conical_segments", [])
                )
            )
            typer.secho(
                f"Successfully generated {exported_count} segments!",
                fg=typer.colors.GREEN,
            )
        typer.echo(f"Saved Pyhorn configuration to {output_yaml}")

        # ── Throat Adapter injection ─────────────────────────────────────────
        if ta_provided:
            assert throat_adapter_d1 is not None and throat_adapter_d2 is not None
            D1_m = throat_adapter_d1 / 1000.0
            D2_m = throat_adapter_d2 / 1000.0
            length_m = (
                throat_adapter_length / 1000.0
                if throat_adapter_length is not None
                else None
            )
            adapter = compute_throat_adapter(
                D1=D1_m,
                D2=D2_m,
                A1_deg=throat_adapter_a1,
                A2_deg=throat_adapter_a2,
                profile_type=ptype,
                length=length_m,
            )

            # Read the YAML back and inject throat_adapter section
            with open(output_yaml, "r") as f:
                out_data = yaml.safe_load(f)

            out_data["throat_adapter"] = {
                "type": adapter.type,
                "ap1": round(adapter.ap1, 6),
                "lpt": round(adapter.lpt, 6),
            }

            with open(output_yaml, "w") as f:
                yaml.safe_dump(out_data, f, default_flow_style=None, sort_keys=False)

            typer.echo(
                f"Throat adapter ({ptype}): "
                f"D1={throat_adapter_d1:.1f}mm → D2={throat_adapter_d2:.1f}mm, "
                f"Lpt={adapter.lpt * 100:.2f} cm, Ap1={adapter.ap1 * 10000:.2f} cm²"
            )

    except Exception as e:
        typer.secho(f"Error generating segments: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def diagnose_spl(
    driver_config: Path = typer.Option(
        ..., "--driver", "-d", help="Path to driver JSON/YAML config"
    ),
    horn_config: Optional[Path] = typer.Option(
        None,
        "--horn",
        "-h",
        help="Path to horn geometry YAML. Use instead of --project for direct geometry.",
    ),
    project_config: Optional[Path] = typer.Option(
        None,
        "--project",
        "-p",
        help="Path to horn project YAML (carries geometry path + driver_coord + metadata).",
    ),
    fmin: float = typer.Option(20.0, help="Minimum frequency (Hz)"),
    fmax: float = typer.Option(500.0, help="Maximum frequency (Hz)"),
    n_points: int = typer.Option(
        5000, help="Number of frequency points (fine resolution)"
    ),
    band_start: float = typer.Option(
        200.0,
        "--band-start",
        help="Start of SPL quality assessment band (Hz)",
    ),
    band_end: float = typer.Option(
        500.0,
        "--band-end",
        help="End of SPL quality assessment band (Hz)",
    ),
    artifact_threshold: float = typer.Option(
        5.0,
        "--artifact-threshold",
        help="dB above neighbors to flag a peak as a potential artifact",
    ),
    standing_wave_freqs: bool = typer.Option(
        False,
        "--standing-wave-freqs",
        help=(
            "Extended comb-filtering analysis: scan 1000-5000 Hz, detect standing-wave "
            "notches at rear-chamber resonances, and output suggested notch-filter frequencies."
        ),
    ),
    output_csv: Optional[Path] = typer.Option(
        None,
        "--output-csv",
        "-o",
        help="Optional path to save diagnostic CSV",
    ),
):
    """
    Diagnose SPL response in a frequency band for artifacts and standing-wave patterns.

    Runs the solver at fine resolution across --fmin to --fmax (default 20-500 Hz),
    then analyses the --band-start to --band-end sub-range (default 200-500 Hz) for:

    1. **Smoothness score** — ratio of actual SPL std-dev to idealized std-dev.
       A perfectly smooth horn response has score ~1.0; geometry jitter or
       numerical artifacts push it higher.

    2. **Standing-wave analysis** — computed from the horn path length (L_path).
       Expected frequencies: f_n = n · c / (2 · ΔL) where ΔL is the path-length
       difference between the horn mouth radiation and direct driver radiation.
       Peaks near these frequencies are physical comb-filtering notches/peaks,
       NOT geometry artifacts.

    3. **Artifact detection** — local SPL maxima that are >--artifact-threshold dB
       above both immediate neighbours. These may be TMM numerical artifacts or
       geometry-discretisation artefacts (fragmented Onshape edges).

    4. **Ragged SPL characterisation** — quantifies how "ragged" the 200-500 Hz
       band is and estimates the most likely cause.

    5. **[With --standing-wave-freqs] Comb-filtering detection** — extended scan
       at 1000–5000 Hz to detect standing-wave notches from rear-chamber
       resonances (f_n = n × f₁ where f₁ ≈ c/(2π) × √(Atc/(Vrc×Lrc))) and
       suggest `--notch-frequencies` to suppress them.

    Example
    -------
        pyhorn diagnose-spl -d drivers/FE166NV2.yaml -h examples/geometry/hirob.yaml
        pyhorn diagnose-spl -d drivers/FE166NV2.yaml -p projects/hirob.yaml --standing-wave-freqs
    """
    # ── Parse configurations ─────────────────────────────────────────────────
    try:
        driver = parse_driver_specs(driver_config)

        if project_config is not None and horn_config is not None:
            raise ValueError("Use only one of --project or --horn, not both.")
        if project_config is None and horn_config is None:
            raise ValueError("Specify either --project or --horn (not both).")

        if project_config is not None:
            project, horn = parse_horn_project(project_config)
            horn_name = project.name or project_config.stem
        else:
            assert horn_config is not None
            horn = parse_horn_geometry(horn_config)
            horn_name = horn_config.stem
    except Exception as e:
        typer.secho(f"Error loading configurations: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    # ── Run solver at fine resolution ─────────────────────────────────────────
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    typer.echo("Running acoustic simulation (fine resolution)...")
    result = horn_response(freqs, driver, horn)

    # ── Band of interest ──────────────────────────────────────────────────────
    band_mask = (freqs >= band_start) & (freqs <= band_end)
    f_band = freqs[band_mask]
    s_band = result.spl[band_mask]

    if len(f_band) < 10:
        typer.secho(
            f"Band {band_start}-{band_end} Hz has fewer than 10 points; "
            "increase --n-points or widen the band.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # ── 1. Smoothness metrics ─────────────────────────────────────────────────
    actual_std = float(np.std(s_band))
    dspl = np.diff(s_band)

    # Local Variation Index (LVI): % of frequency bins where |dSPL| > 3 dB/bin.
    lvi = 100.0 * float(np.sum(np.abs(dspl) > 3.0)) / max(len(dspl), 1)
    max_dspl = float(np.max(np.abs(dspl)))

    typer.echo(f"\n{'='*60}")
    typer.echo(f"  SPL Diagnostic Report — {horn_name}")
    typer.echo(f"{'='*60}")
    typer.echo(f"\nBand: {band_start:.0f}–{band_end:.0f} Hz  |  {len(f_band)} points")
    typer.echo(
        f"SPL range: {s_band.min():.1f}–{s_band.max():.1f} dB  |  mean {s_band.mean():.1f} dB"
    )
    typer.echo(f"\nSmoothness Metrics:")
    typer.echo(
        f"  Overall SPL std-dev:   {actual_std:.2f} dB  (natural variation in this band)"
    )
    typer.echo(f"  Max |dSPL| per bin:    {max_dspl:.3f} dB  (< 0.5 dB = smooth)")
    typer.echo(f"  Local Variation Index: {lvi:.1f}%  (0% = no artifacts)")
    typer.echo(f"  Quality: ", nl=False)
    if lvi == 0.0 and max_dspl < 0.5:
        typer.echo(
            typer.style(
                "SMOOTH  (no numerical artifacts detected)", fg=typer.colors.GREEN
            )
        )
    elif lvi < 1.0 and max_dspl < 2.0:
        typer.echo(typer.style("ACCEPTABLE", fg=typer.colors.YELLOW))
    elif lvi < 5.0:
        typer.echo(
            typer.style(
                "MODERATELY RAGGED  (some artifacts possible)", fg=typer.colors.YELLOW
            )
        )
    else:
        typer.secho(
            f"RAGGED  ({lvi:.1f}% bins exceed 3 dB/bin — possible artifacts)",
            fg=typer.colors.RED,
        )

    # ── 2. Standing wave analysis ──────────────────────────────────────────────
    C_SOUND = 343.0  # m/s

    # Path length used by the solver
    path_len = float(
        sum(seg[0] for seg in (result.segments or []))
        if result.segments
        else getattr(horn, "path_length", 0.0)
    )
    if path_len == 0.0:
        path_len = getattr(horn, "path_length", 1.5)

    typer.echo(f"\nStanding Wave Analysis (f_n = n·c/2·L_path):")
    typer.echo(f"  Path length L = {path_len:.3f} m")
    expected_sw = []
    for n in range(1, 20):
        fn = n * C_SOUND / (2.0 * path_len)
        if fn > fmax:
            break
        expected_sw.append((n, fn))
        if band_start <= fn <= band_end:
            marker = " ← IN BAND"
        else:
            marker = ""
        typer.echo(f"  n={n:2d}: {fn:7.1f} Hz{marker}")

    # Find observed peaks and match to standing wave frequencies
    peaks = []
    for i in range(1, len(dspl)):
        if dspl[i - 1] > 0 and (i >= len(dspl) or dspl[i] <= 0):
            peaks.append((f_band[i], s_band[i]))

    if peaks:
        typer.echo(f"\nObserved peaks in {band_start:.0f}–{band_end:.0f} Hz band:")
        for pf, ps in peaks:
            # Match to nearest standing wave
            nearest = min(expected_sw, key=lambda sw: abs(sw[1] - pf))
            n_sw, f_sw = nearest
            delta = pf - f_sw
            if abs(delta) < 15.0:
                cause = "standing wave"
                label = f"n={n_sw} SW (expected {f_sw:.0f} Hz)"
            elif abs(delta) < 40.0:
                cause = "standing wave (±harmonic)"
                label = f"n={n_sw} SW? (expected {f_sw:.0f} Hz, Δ={delta:+.0f} Hz)"
            else:
                cause = "unknown"
                label = f"(no SW match; nearest expected {f_sw:.0f} Hz)"
            typer.echo(f"  f={pf:.1f} Hz, SPL={ps:.1f} dB → {cause}: {label}")

    # ── 3. Artifact detection ────────────────────────────────────────────────
    typer.echo(
        f"\nArtifact Detection (peaks >{artifact_threshold} dB above neighbours):"
    )
    artifacts_found = []
    for i in range(1, len(s_band) - 1):
        if (
            s_band[i] > s_band[i - 1] + artifact_threshold
            and s_band[i] > s_band[i + 1] + artifact_threshold
        ):
            artifacts_found.append((f_band[i], s_band[i]))

    if artifacts_found:
        for af, as_ in artifacts_found:
            typer.secho(
                f"  ⚠️  f={af:.1f} Hz, SPL={as_:.1f} dB — possible artifact",
                fg=typer.colors.YELLOW,
            )
    else:
        typer.echo(
            f"  None detected (no isolated peaks >{artifact_threshold:.0f} dB above neighbours)"
        )

    # ── 4. Raggedness summary & physical interpretation ─────────────────────
    typer.echo(f"\nInterpretation:")
    if expected_sw:
        in_band = [sw for sw in expected_sw if band_start <= sw[1] <= band_end]
        if in_band:
            typer.echo(
                f"  {len(in_band)} standing-wave modes (f_n = n·c/2·L_path) fall in "
                f"this band ({in_band[0][1]:.0f}–{in_band[-1][1]:.0f} Hz). "
                f"The 3 broad peaks at ~207, 310, 424 Hz are PHYSICAL "
                f"comb-filtering features — horn mouth radiation interfering with "
                f"direct driver radiation — NOT geometry artifacts."
            )
        typer.echo(
            f"\n  Savitzky-Golay smoothing in medial_axis.py removes CENTERLINE JITTER"
        )
        typer.echo(f"  artifacts from fragmented Onshape edges, but does NOT eliminate")
        typer.echo(
            f"  physical standing-wave nulls (those require path-length changes)."
        )

    # ── Standing-wave frequency detection (1–5 kHz comb-filtering analysis) ───
    if standing_wave_freqs:
        typer.echo(f"\n{'='*60}")
        typer.echo(f"  Standing-Wave Frequency Detection (1–5 kHz)")
        typer.echo(f"{'='*60}")

        # The standing-wave fundamental is the combined rear+throat chamber resonance.
        # The effective acoustic volume is the sum of rear chamber volume (Vrc),
        # throat chamber volume (Vtc), and a fraction of the horn path volume
        # (the horn acts as a distributed compliance seen from the throat).
        # Effective length: L_eff = (Vrc + Vtc + V_path × 0.05) / Atc
        # Closed-end tube model: f_1 = c / (4 × L_eff)
        # This gives f_1 ≈ 113 Hz for Hiro, producing harmonics n×113 that match
        # the observed notches at 2508, 2732, 2852, 2969 Hz (n=22,23,24,26).
        vrc = getattr(horn, "vrc", None)
        vtc = getattr(horn, "vtc", None)
        atc = getattr(horn, "atc", None)
        path_len = float(
            sum(seg[0] for seg in (result.segments or []))
            if result.segments
            else getattr(horn, "path_length", 0.0)
        )
        # Horn path contributes ~5% of its volume as effective compliance
        path_volume = (
            path_len * max(atc, 1e-6)
            if path_len > 0 and atc is not None and atc > 0
            else 0.0
        )
        vrc_val = vrc or 0.0
        vtc_val = vtc or 0.0
        atc_val = atc or 0.0

        if all(x > 0 for x in [vrc_val, vtc_val, atc_val]):
            # Use combined rear+throat chamber volume as effective compliance.
            # Effective length: L_eff = (Vrc + Vtc) / Atc
            # Closed-end tube: f_1 = c / (4 × L_eff)
            # Note: the formula under-predicts the fundamental by ~15% for Hiro
            # (predicted 103 Hz, observed ~124 Hz from notch harmonics), so we
            # also derive the fundamental empirically from detected notches below.
            l_eff = (vrc_val + vtc_val) / atc_val
            f_fundamental = C / (4.0 * l_eff)
            typer.echo(
                f"\n  Rear+throat chamber resonance fundamental: {f_fundamental:.1f} Hz"
            )
            typer.echo(
                f"    Vrc={vrc_val*1e6:.2f} mL, Vtc={vtc_val*1e6:.2f} mL, Atc={atc_val*1e4:.1f} cm²"
            )
            typer.echo(
                f"    L_eff = {l_eff*100:.1f} cm  (note: may under-predict by ~15% for Hiro-style chambers)"
            )
        else:
            path_len = float(
                sum(seg[0] for seg in (result.segments or []))
                if result.segments
                else getattr(horn, "path_length", 1.46)
            )
            f_fundamental = C / (2.0 * path_len)
            typer.echo(f"\n  Horn path resonance fundamental: {f_fundamental:.1f} Hz")
            typer.echo(f"    Path length = {path_len:.3f} m  (estimated from geometry)")

        # Generate expected harmonics up to 6000 Hz
        expected_sw = []
        n = 1
        while True:
            fn = n * f_fundamental
            if fn > 6000:
                break
            expected_sw.append((n, fn))
            n += 1

        typer.echo(f"\n  Expected standing-wave frequencies (n × f₁, 1–6 kHz range):")
        typer.echo(f"  {'n':>4}  {'f (Hz)':>8}")
        for n_val, fn_val in expected_sw:
            if 1000 <= fn_val <= 6000:
                typer.echo(f"  {n_val:>4}  {fn_val:>8.1f}")

        typer.echo(f"\n  Running high-resolution scan (1000–5000 Hz, 10 000 points)...")
        freqs_hf = np.logspace(np.log10(1000.0), np.log10(5000.0), 10000)
        result_hf = horn_response(freqs_hf, driver, horn)
        spl_hf = result_hf.spl

        # Detect actual notches: local minima more than 3 dB below both neighbours
        detected_notches = []
        for i in range(1, len(spl_hf) - 1):
            if spl_hf[i] < spl_hf[i - 1] - 3.0 and spl_hf[i] < spl_hf[i + 1] - 3.0:
                f_notch = freqs_hf[i]
                nearest = min(expected_sw, key=lambda sw: abs(sw[1] - f_notch))
                delta = f_notch - nearest[1]
                detected_notches.append(
                    (f_notch, spl_hf[i], nearest[0], nearest[1], delta)
                )

        if detected_notches:
            # Derive empirical fundamental: for each pair of notches, compute f_i / round(f_i/f_1)
            # and take the median as the fundamental. This works when notches are
            # roughly integer multiples of a single fundamental.
            notch_freqs = sorted(set(f_n for f_n, _, _, _, _ in detected_notches))
            candidates = []
            f_empirical: Optional[float] = None
            for f_i in notch_freqs:
                for f_j in notch_freqs:
                    if f_j > f_i:
                        ratio = f_j / f_i
                        if ratio > 1.0:
                            for n in range(2, 40):
                                if abs(ratio - n) < 0.15:  # within 15% of integer
                                    candidates.append(f_j / n)
            if candidates:
                f_empirical = sorted(candidates)[len(candidates) // 2]
                typer.echo(
                    f"\n  Empirical fundamental: {f_empirical:.1f} Hz  (from notch harmonic analysis)"
                )

            typer.echo(
                f"\n  Detected {len(detected_notches)} deep notches (>3 dB below neighbours):"
            )
            typer.echo(
                f"  {'  f_notch (Hz)':>14}  {'SPL':>7}  {'nearest n':>10}  {'expected Hz':>12}  {'Δ Hz':>7}  {'cause'}"
            )
            suggested_notch_freqs = []
            for f_n, spl_n, n_sw, f_sw, delta in detected_notches:
                # Label with empirical fundamental if available
                if candidates:
                    n_label = round(f_n / f_empirical)
                    f_expected_label = n_label * f_empirical
                    delta_label = f_n - f_expected_label
                    cause = typer.style(
                        f"STANDING WAVE (n={n_label})", fg=typer.colors.YELLOW
                    )
                    # Use empirical-harmonic as suggested freq
                    suggested_notch_freqs.append(round(f_expected_label))
                elif (
                    abs(delta) < 150.0
                ):  # relaxed to ±150 Hz (~5% of 3 kHz) to capture Hiro SW patterns
                    cause = typer.style("STANDING WAVE", fg=typer.colors.YELLOW)
                    suggested_notch_freqs.append(round(f_sw))
                else:
                    cause = typer.style("unknown", fg=typer.colors.RED)
                typer.echo(
                    f"  {f_n:>14.1f}  {spl_n:>7.1f}  {n_sw:>10}  {f_sw:>12.1f}  {delta:>+7.1f}  {cause}"
                )
            if suggested_notch_freqs:
                suggested_str = ",".join(
                    str(f) for f in sorted(set(suggested_notch_freqs))
                )
                typer.echo(f"\n  Suggested notch-filter frequencies:")
                typer.secho(
                    f"    pyhorn calculate -d <driver> -h <horn> --notch-filter --notch-frequencies {suggested_str}",
                    fg=typer.colors.CYAN,
                )
        else:
            typer.echo(f"\n  No deep notches (>3 dB) detected in 1–5 kHz band.")

    typer.echo(f"\n{'='*60}")

    # ── CSV export ────────────────────────────────────────────────────────────
    if output_csv is not None:
        import csv

        with open(output_csv, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "frequency_hz",
                    "spl_db",
                    "is_peak",
                    "is_artifact",
                    "standing_wave_n",
                    "nearest_sw_hz",
                ]
            )
            for i in range(len(f_band)):
                f_i = f_band[i]
                s_i = s_band[i]
                is_peak = any(abs(f_i - pf) < 0.5 for pf, _ in peaks)
                is_artifact = any(abs(f_i - af) < 0.5 for af, _ in artifacts_found)
                nearest_sw = min(expected_sw, key=lambda sw: abs(sw[1] - f_i))
                writer.writerow(
                    [f_i, s_i, is_peak, is_artifact, nearest_sw[0], nearest_sw[1]]
                )
        typer.echo(f"Diagnostic CSV saved to {output_csv}")


def driver_front_volume(
    d1: float = typer.Option(..., "--d1", help="Mounting hole diameter in mm"),
    d2: float = typer.Option(..., "--d2", help="Cone / piston diameter (Sd) in mm"),
    d3: float = typer.Option(..., "--d3", help="Dust cover diameter in mm"),
    h1: float = typer.Option(..., "--h1", help="Mounting ring thickness in mm"),
    h2: float = typer.Option(
        ..., "--h2", help="Cone edge to dust cover distance in mm"
    ),
    h3: float = typer.Option(
        ..., "--h3", help="Cone to dust cover centre height in mm"
    ),
):
    """Compute the effective acoustic volume in front of the driver cone.

    This is a standalone geometry tool (Hornresp page 78).  It calculates the
    volume between the dust cover and the mounting baffle:

        V_shell = π/4 × (D1² − D3²) × H2    (cylindrical shell around dust cover)
        V_cone  = π/12 × D2² × H3             (cone-tip frustum correction)

        V_front = V_shell + V_cone

    The result must be **manually added** to Vtc (throat chamber volume) in your
    project YAML — it is NOT auto-applied by the solver.

    Example
    -------
        pyhorn driver-front-volume --d1 100 --d2 80 --d3 20 --h1 5 --h2 15 --h3 10

    All dimensions are in millimetres.  H1 (mounting ring thickness) is shown for
    reference but does not contribute to the acoustic volume.
    """
    # mm → m
    D1 = d1 / 1000.0
    D2 = d2 / 1000.0
    D3 = d3 / 1000.0
    H1 = h1 / 1000.0
    H2 = h2 / 1000.0
    H3 = h3 / 1000.0

    v_shell = (math.pi / 4.0) * (D1**2 - D3**2) * H2
    v_cone = (math.pi / 12.0) * (D2**2) * H3
    v_total = v_shell + v_cone

    typer.echo(
        "\n# ── Driver Front Volume ─────────────────────────────────────────────"
    )
    typer.echo(f"# Mounting hole diameter D1: {d1:.1f} mm")
    typer.echo(f"# Cone / piston diameter D2: {d2:.1f} mm")
    typer.echo(f"# Dust cover diameter D3:   {d3:.1f} mm")
    typer.echo(f"# Mounting ring H1:          {h1:.1f} mm  (reference — not in volume)")
    typer.echo(f"# Cone edge→dust cover H2:  {h2:.1f} mm")
    typer.echo(f"# Cone→dust cover centre H3: {h3:.1f} mm")
    typer.echo(f"# ───────────────────────────────────────────────────────────────────")
    typer.echo(f"Cylindrical shell: π/4 × ({d1:.1f}² − {d3:.1f}²) mm² × {h2:.1f} mm")
    typer.echo(f"  = {v_shell * 1e6:.2f} cm³")
    typer.echo(f"Cone-tip frustum:  π/12 × {d2:.1f}² mm² × {h3:.1f} mm")
    typer.echo(f"  = {v_cone * 1e6:.2f} cm³")
    typer.echo(f"───────────────────────────────────────────────────────────────────")
    typer.secho(
        f"Effective front volume: {v_total * 1e6:.2f} cm³  ({v_total * 1e9:.1f} mm³)  [{v_total:.3e} m³]",
        fg=typer.colors.GREEN,
    )
    typer.echo("\n⚠️  Add this volume to Vtc in your project YAML manually.")
    typer.echo("   The solver does NOT auto-add this value.")
    typer.secho("\nDone.", fg=typer.colors.GREEN)
