"""Core simulation commands: calculate, compare, derive-ts."""

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
from pyhorn_core.config.horn_models import HornGeometry
from pyhorn_core.config.project_models import HornProject
from pyhorn_core.solver.models import horn_response, RHO, C
from pyhorn_core.solver.profiles import _solve_hyperbolic_u
from pyhorn_core.output.plotter import (
    plot_simulation_results,
    plot_waterfall,
    plot_impulse_step,
)
from pyhorn_core.solver.spectrogram import compute_spectrogram, plot_spectrogram
from pyhorn_core.solver.time_domain import compute_csd, export_impulse_to_wav
from pyhorn_core.output.exporter import export_to_csv, export_to_json, export_to_frd
from pyhorn_core.solver.room import compute_room_gain

from ._shared import (
    _compute_piston_off_axis_spl,
    _print_radiation_summary,
    _folded_throat_chamber_side,
)


def calculate(
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
    output_dir: Path = typer.Option(
        Path("./outputs"), "--output-dir", "-o", help="Directory to save outputs"
    ),
    fmin: float = typer.Option(20.0, help="Minimum frequency (Hz)"),
    fmax: float = typer.Option(5000.0, help="Maximum frequency (Hz)"),
    n_points: int = typer.Option(500, help="Number of frequency points"),
    export_csv: bool = typer.Option(True, help="Export data to CSV"),
    export_json: bool = typer.Option(False, help="Export data to JSON"),
    export_wav: Optional[Path] = typer.Option(
        None,
        "--export-wav",
        help="Export impulse response as a 16-bit PCM WAV file at this path. "
        "Reconstructs impulse response from SPL + phase via IFFT (Hornresp page 96).",
    ),
    export_frd: Optional[Path] = typer.Option(
        None,
        "--export-frd",
        help="Export frequency response data as a .frd file (REW/ARTA format). "
        "Contains magnitude (dB SPL) and phase (degrees) at log-spaced frequencies.",
    ),
    plot: bool = typer.Option(True, help="Generate SPL plot (.png)"),
    plot_phase: bool = typer.Option(
        True,
        "--plot-phase/--no-plot-phase",
        help="Include phase (degrees) + group delay (ms) and throat acoustic impedance panels in response_plot.png",
    ),
    plot_distortion: bool = typer.Option(
        True,
        "--plot-distortion/--no-plot-distortion",
        help="Include second tone distortion panel (SPL(2f) - SPL(f) dB below fundamental) in response_plot.png. "
        "Only computed for single-segment horns.",
    ),
    spl_only: bool = typer.Option(
        False,
        "--spl-only",
        help="Generate a single-panel SPL plot only (no impedance, excursion, or other panels).",
    ),
    distortion: bool = typer.Option(
        True,
        "--distortion/--no-distortion",
        help="Compute second tone distortion and include it in the CSV/JSON export. "
        "Only for single-segment horns; adds 'Second tone distortion (dB rel)' column.",
    ),
    plot_3d: bool = typer.Option(True, help="Generate 3D horn schematic (.png)"),
    target_spl: Optional[float] = typer.Option(
        None, help="Target SPL for reference line (defaults to driver reference SPL)"
    ),
    target_impedance: Optional[float] = typer.Option(
        None, help="Target impedance line in Ohms (defaults to driver Re)"
    ),
    target_excursion: Optional[float] = typer.Option(
        None,
        help="Target excursion line in mm peak (defaults to driver xmax if provided)",
    ),
    output_mode: str = typer.Option(
        "combined",
        "--output-mode",
        help="SPL output mode: 'combined' (total, default), 'horn' (horn-only), or 'element' (direct radiator only)",
    ),
    off_axis_angles: Optional[str] = typer.Option(
        None,
        "--off-axis-angles",
        help="Comma-separated off-axis angles in degrees for directivity export "
        "(e.g. 0,15,30,45,60,75,90). Computed using Levine/Inglis piston model "
        "at the horn mouth. Omit to skip off-axis export.",
    ),
    polar_freq: Optional[float] = typer.Option(
        1000.0,
        "--polar-freq",
        help="Add a polar directivity panel to response_plot.png at this frequency (Hz). "
        "Default: 1000 Hz. Pass 0 to disable the polar panel.",
    ),
    radiation_summary: bool = typer.Option(
        False,
        "--radiation-summary",
        help="Print a radiation summary table after the main output: on-axis SPL, "
        "-6dB beamwidth, and directivity index at 250, 500, 1000, 2000, and 4000 Hz. "
        "Uses Levine/Inglis piston model at the horn mouth.",
    ),
    voice_coil_temp: Optional[float] = typer.Option(
        None,
        "--voice-coil-temp",
        help="Voice coil temperature in °C for thermal power compression modeling. "
        "Default: no thermal modeling. Typical sustained-operation value: 100°C. "
        "Reference: Hornresp page 98 (thermal power compression expressed in decibels).",
    ),
    thermal_compression: bool = typer.Option(
        False,
        "--thermal-compression",
        help="Include thermal power compression data in the CSV/JSON export "
        "and print a summary. Only active when --voice-coil-temp is also provided.",
    ),
    notch_filter: bool = typer.Option(
        False,
        "--notch-filter",
        help="Apply narrow IIR notch filters at the specified artifact frequencies "
        "to suppress TMM numerical artifacts (e.g. the ~1847 Hz Hiro resonance). "
        "The notched SPL is stored in result.spl_notched alongside the raw SPL.",
    ),
    notch_frequencies: Optional[str] = typer.Option(
        None,
        "--notch-frequencies",
        help="Comma-separated list of centre frequencies in Hz for notch filters. "
        "Example: --notch-frequencies 1847,2508,2732,2852,2969 "
        "(the Hiro-project characterised artifact frequencies). "
        "Defaults to these values when --notch-filter is active and this flag is omitted.",
    ),
    notch_q: float = typer.Option(
        10.0,
        "--notch-q",
        help="Quality factor for notch filters. Higher Q means narrower notch. "
        "Q=10 gives approximately 10% bandwidth at the -3dB points. Default: 10.0.",
    ),
    filter_delay_mode: str = typer.Option(
        "group_delay",
        "--filter-delay-mode",
        help="Group delay display mode for Filter Wizard UI: "
        "'group_delay' (standard, ms) or 'per_period' (dimensionless = τ_g × f). "
        "The per-period variant normalises delay by wavelength for octave-comparable values. "
        "Hornresp page 120 — Filter Wizard 'Delay' option.",
    ),
    fdd: bool = typer.Option(
        False,
        "--fdd",
        help="Use the FDD (Frequency Dependent Directivity) model instead of the "
        "standard piston (Levine/Inglis) model for off-axis directivity computation. "
        "The FDD model provides a smooth transition from omnidirectional radiation "
        "at low frequencies to increasingly directional (narrowing beamwidth) radiation "
        "at high frequencies. "
        "Reference: Hornresp pages 77 and 92.",
    ),
    fdd_fc: float = typer.Option(
        300.0,
        "--fdd-fc",
        help="FDD characteristic transition frequency in Hz (f_c). "
        "At f = f_c the directivity index is D_max × (1 − e⁻¹) ≈ 0.63·D_max. "
        "At f = 2·f_c it is ≈ 0.98·D_max. "
        "Typical values: 200–400 Hz for mid-size horn mouths. Default: 300 Hz. "
        "Only active when --fdd is used.",
    ),
    fdd_dmax: float = typer.Option(
        5.0,
        "--fdd-dmax",
        help="FDD maximum directivity index in dB (D_max). "
        "The asymptotic high-frequency DI. "
        "Typical values: 3–6 dB for mid-size horn mouths, 6–10 dB for large "
        "or multi-segment mouths. Default: 5.0 dB. "
        "Only active when --fdd is used.",
    ),
    room_type: str = typer.Option(
        "free_space",
        "--room-type",
        help="Room boundary type for power response gain (Hornresp page 96). "
        "Options: 'free_space' (no gain), 'half_space' (+3 dB near walls), "
        "'quarter_space' (+6 dB in corner), 'eighth_space' (+9 dB bookshelf/recess). "
        "The gain rolls off as 1/f² above the room-mode cutoff frequency.",
    ),
    room_volume: Optional[float] = typer.Option(
        None,
        "--room-volume",
        help="Room volume in cubic metres for Sabine room-mode estimation. "
        "Used with --room-type to refine the room gain cutoff frequency.",
    ),
    futtrup_gd: bool = typer.Option(
        False,
        "--futtrup-gd/--no-futtrup-gd",
        help="Include the Futtrup audible group delay limit curve in the CSV/JSON export. "
        "The Futtrup formula (Hornresp page 113) defines the audible GD ceiling: "
        "GDlimit = 1000 × 1160.6 / (5643 × f^0.81511 − f) ms. "
        "Horn responses whose group delay exceeds this limit are typically inaudible "
        "due to precedence masking.",
    ),
    spectrogram: bool = typer.Option(
        False,
        "--spectrogram",
        help="Generate a standalone spectrogram PNG showing spectral intensity vs. "
        "frequency and time (ms). Uses STFT of the impulse response. "
        "Implies --plot.",
    ),
    spectrogram_window_ms: float = typer.Option(
        50.0,
        "--spectrogram-window-ms",
        help="STFT window duration in milliseconds for --spectrogram. "
        "Smaller window gives better time resolution. Default: 50 ms.",
    ),
    spectrogram_overlap: float = typer.Option(
        0.5,
        "--spectrogram-overlap",
        help="STFT window overlap fraction for --spectrogram (0–1). "
        "Higher overlap gives smoother time axis. Default: 0.5 (50%).",
    ),
    filter_schematic: bool = typer.Option(
        False,
        "--filter-schematic",
        help="Print the filter schematic ASCII art and exit. "
        "Use --filter-preset or --filter-yaml to specify the filter bands. "
        "No simulation is run.",
    ),
    filter_preset: Optional[str] = typer.Option(
        None,
        "--filter-preset",
        help="Filter preset name to show schematic for. "
        "Available presets: "
        "2-way crossover (LR2 12dB/oct), "
        "3-way crossover (LR2 12dB/oct), "
        "peaking EQ (+3dB at 2.5kHz), "
        "high-shelf cut (-3dB above 4kHz), "
        "low-shelf boost (+3dB below 200Hz), "
        "notch filter (-12dB at 1kHz), "
        "Le Cleach HP. "
        "Use --filter-schematic without this flag for the Le Cleach default.",
    ),
    filter_yaml: Optional[Path] = typer.Option(
        None,
        "--filter-yaml",
        help="Path to a filter YAML config file. "
        "With --filter-schematic: print the filter schematic and exit. "
        "Without --filter-schematic: apply filter bands as post-processing. "
        "See also: --filter (shorthand for apply-only mode).",
    ),
    filter_path: Optional[Path] = typer.Option(
        None,
        "--filter",
        help="Path to a filter YAML config file. "
        "Reads filter_bands from the YAML and applies them as post-processing "
        "to the horn response (adds Filtered SPL, Filtered Impedance, "
        "Filtered Phase, and Filter contribution dB columns to the export). "
        "Example: --filter filters/my_hp_200hz.yaml",
    ),
    benchmark: bool = typer.Option(
        False,
        "--benchmark",
        help="Run the canonical HiroB Hornresp benchmark fixture. "
        "Automatically loads tests/benchmarks/hornresp/hirob/fixture/horn.yaml "
        "and prints a comparison banner.",
    ),
    benchmark_project: Optional[Path] = typer.Option(
        None,
        "--benchmark-project",
        help="Path to a custom benchmark horn YAML. "
        "Implies --benchmark when provided.",
    ),
    path_length_diff: Optional[float] = typer.Option(
        None,
        "--path-length-diff",
        help="Listening distance offset in metres for finite horn-charged bass reflex. "
        "Positive values add a frequency-dependent phase lag Δφ = 2π·pld/c·f to the port "
        "radiation before summing with the horn output (Hornresp page 91). "
        "Overrides the value in the project YAML if provided.",
    ),
):
    """
    Calculate the acoustic response of a horn enclosure and save the results.

    Use --project (-p) to load a project YAML that references the geometry and
    carries metadata (driver_coord, name, notes, plot overrides) — this is the
    recommended way to run simulations.  Alternatively, use --horn (-h) to point
    directly at a geometry YAML.
    """
    # ── 0. Filter Schematic (no horn config needed) ───────────────────────────
    if filter_schematic:
        from pyhorn_core.solver.filter_schematic import (
            compute_filter_schematic,
            FilterBand,
        )

        if filter_yaml is not None:
            try:
                with open(filter_yaml) as fh:
                    filter_data = yaml.safe_load(fh)
                band_list = (
                    filter_data.get("filter_bands", [])
                    if isinstance(filter_data, dict)
                    else []
                )
                bands: list[FilterBand] = [
                    FilterBand(
                        type=b.get("type", "peakingEQ"),
                        frequency=float(b.get("frequency", 1000)),
                        q=float(b.get("q", 1.0)),
                        gain_db=float(b.get("gain_db", 0.0)),
                        order=int(b.get("order", 2)),
                        enabled=b.get("enabled", True),
                    )
                    for b in band_list
                ]
            except Exception as exc:
                typer.secho(f"Error reading filter YAML: {exc}", fg=typer.colors.RED)
                raise typer.Exit(code=1)
        elif filter_preset is not None:
            from pyhorn_ui.server import _DEFAULT_BANDS

            preset_bands = _DEFAULT_BANDS.get(filter_preset)
            if preset_bands is None:
                typer.secho(
                    f"Unknown filter preset '{filter_preset}'. "
                    f"Available: {', '.join(_DEFAULT_BANDS.keys())}",
                    fg=typer.colors.RED,
                )
                raise typer.Exit(code=1)
            bands = [
                FilterBand(
                    type=b.type,
                    frequency=b.frequency,
                    q=b.q,
                    gain_db=b.gain_db,
                    order=b.order,
                    enabled=b.enabled,
                )
                for b in preset_bands
            ]
        else:
            # Default: Le Cleach HP preset
            from pyhorn_ui.server import _DEFAULT_BANDS

            preset_bands = _DEFAULT_BANDS.get(
                "Le Cleach HP", _DEFAULT_BANDS.get("Le Cléac'h HP", [])
            )
            bands = [
                FilterBand(
                    type=b.type,
                    frequency=b.frequency,
                    q=b.q,
                    gain_db=b.gain_db,
                    order=b.order,
                    enabled=b.enabled,
                )
                for b in preset_bands
            ]

        typer.echo(compute_filter_schematic(bands))
        raise typer.Exit(code=0)

    # ── Benchmark mode ─────────────────────────────────────────────────────────
    if benchmark or benchmark_project is not None:
        benchmark_proj_path = benchmark_project
        if benchmark_proj_path is None:
            # Default: use the canonical HiroB benchmark horn fixture
            benchmark_proj_path = (
                Path(__file__).parent.parent.parent
                / "tests"
                / "benchmarks"
                / "hornresp"
                / "hirob"
                / "fixture"
                / "horn.yaml"
            )
        typer.secho(
            "\n"
            "╔══════════════════════════════════════════════════════════════╗\n"
            "║  Running benchmark: HiroB Hornresp comparison              ║\n"
            "║  Reference: tests/benchmarks/hornresp/hirob/reference     ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n",
            fg=typer.colors.CYAN,
        )
        typer.echo(f"  Benchmark horn: {benchmark_proj_path}")
        if project_config is None and horn_config is None:
            horn_config = benchmark_proj_path
            typer.echo("  Auto-setting --horn from benchmark fixture.")
        else:
            typer.secho(
                "  Note: --project or --horn already specified; using those instead.",
                fg=typer.colors.YELLOW,
            )

    # ── 1. Parse configurations ─────────────────────────────────────────────────
    horn: Optional[HornGeometry] = None
    project: Optional[HornProject] = None
    horn_name: str = ""
    try:
        driver = parse_driver_specs(driver_config)

        if project_config is not None and horn_config is not None:
            raise ValueError("Use only one of --project or --horn, not both.")
        if project_config is None and horn_config is None:
            raise ValueError("Specify either --project or --horn.")

        if project_config is not None:
            project, horn = parse_horn_project(project_config)
            horn_name = project.name or project_config.stem
            if project.notes:
                typer.echo(f"Project: {project.notes}")
            typer.echo(f"Loaded project from {project_config}")
            typer.echo(f"  geometry_path: {project.geometry_path}")
            if project.driver_coord:
                typer.echo(f"  driver_coord: {project.driver_coord}")
        elif horn_config is not None:
            horn = parse_horn_geometry(horn_config)
            horn_name = horn_config.stem
            typer.echo(f"Loaded driver specs from {driver_config}")
            typer.echo(f"Loaded horn geometry from {horn_config}")

    except Exception as e:
        typer.secho(f"Error loading configurations: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    assert horn is not None

    # ── Override path_length_difference (CLI flag takes precedence over YAML) ──
    if (
        path_length_diff is not None
        and horn is not None
        and horn.vented_box is not None
    ):
        horn.vented_box.path_length_difference = path_length_diff
        typer.echo(
            f"  path_length_difference set to {path_length_diff:.3f} m (CLI override)"
        )

    # 2. Prepare Frequency Array
    freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_points)

    # Validate output_mode
    valid_modes = {"combined", "horn", "element"}
    if output_mode not in valid_modes:
        typer.secho(
            f"Invalid --output-mode '{output_mode}'. Must be one of: {', '.join(sorted(valid_modes))}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    # ── Acoustic Simulation (skip in wavefront-only mode) ──────────────────────
    run_acoustic_sim = horn is not None

    if run_acoustic_sim:
        # ── Notch filter parsing ───────────────────────────────────────────────
        _notch_freqs: Optional[list] = None
        if notch_filter:
            if notch_frequencies:
                try:
                    _notch_freqs = [
                        float(f.strip()) for f in notch_frequencies.split(",")
                    ]
                except ValueError:
                    typer.secho(
                        f"Error: could not parse --notch-frequencies '{notch_frequencies}'. "
                        "Use comma-separated numbers, e.g. 1847,2508,2732,2852,2969.",
                        fg=typer.colors.RED,
                    )
                    raise typer.Exit(code=1)
            else:
                # Default Hiro-characterised artifact frequencies
                _notch_freqs = [1847.0, 2508.0, 2732.0, 2852.0, 2969.0]
                typer.echo(
                    f"Using default notch frequencies: {_notch_freqs} Hz (Hiro characterisation)"
                )

        typer.echo("Running acoustic simulation...")
        if fdd:
            typer.echo(f"  FDD mode: fc={fdd_fc:.0f} Hz, D_max={fdd_dmax:.1f} dB")
        result = horn_response(
            freqs,
            driver,
            horn,
            T_voice=voice_coil_temp,
            compute_distortion=distortion,
            notch_filter=notch_filter,
            notch_frequencies=_notch_freqs,
            notch_q=notch_q,
            fdd_mode=fdd,
            fdd_fc=fdd_fc,
            fdd_dmax=fdd_dmax,
        )

        # ── Room boundary gain (Hornresp page 96) ──────────────────────────────
        if room_type != "free_space":
            room_gain_db = compute_room_gain(
                freqs,
                room_type,
                distance_to_wall_m=None,
                room_volume_m3=room_volume,
            )
            result.room_gain_db = room_gain_db
            result.room_type = room_type
            # Print summary
            max_gain = float(np.max(room_gain_db))
            f_max_gain = float(freqs[np.argmax(room_gain_db)])
            typer.echo(
                f"  Room gain: {room_type} — peak {max_gain:.1f} dB at "
                f"{f_max_gain:.0f} Hz, rolls off above ~300 Hz"
            )
        else:
            result.room_gain_db = np.zeros_like(freqs, dtype=float)
            result.room_type = "free_space"

    # Output directory
    horn_output_dir = output_dir / horn_name
    horn_output_dir.mkdir(parents=True, exist_ok=True)

    if run_acoustic_sim:
        # ── Thermal power compression warning ──────────────────────────────────
        if voice_coil_temp is not None and voice_coil_temp > 80.0:
            alpha = getattr(driver, "alpha_re", 0.00393)
            re_hot = driver.re * (1.0 + alpha * (voice_coil_temp - 20.0))
            typer.secho(
                f"Thermal compression: Re increases to {re_hot:.2f} Ω at "
                f"T={voice_coil_temp:.0f}°C (nominal Re={driver.re:.2f} Ω at 20°C)",
                fg="yellow",
            )
            if result.thermal_compression_db is not None:
                typer.secho(
                    f"  Compression dB: "
                    f"avg={np.mean(result.thermal_compression_db):.2f}, "
                    f"max={np.min(result.thermal_compression_db):.2f} dB "
                    f"(most compressed frequency)",
                    fg="yellow",
                )

    # 4. Compute off-axis SPL if requested (piston directivity at horn mouth)
    off_axis_spl: Optional[dict] = None
    if off_axis_angles is not None:
        angles_list = [int(a.strip()) for a in off_axis_angles.split(",") if a.strip()]
        if horn.mouth_area is None or horn.mouth_area <= 0:
            typer.secho(
                " --off-axis-angles requires a valid horn.mouth_area; skipping off-axis export.",
                fg=typer.colors.YELLOW,
            )
        else:
            off_axis_spl = _compute_piston_off_axis_spl(
                freqs, horn.mouth_area, angles_list
            )
            typer.echo(f"  Off-axis angles: {angles_list} (piston model at mouth)")

    # 5. Determine the primary SPL for the selected output mode
    if output_mode == "horn":
        primary_spl = result.horn_spl if result.horn_spl is not None else result.spl
        if result.horn_spl is None:
            typer.secho(
                "output-mode=horn but no horn SPL available — using combined SPL.",
                fg=typer.colors.YELLOW,
            )
    elif output_mode == "element":
        primary_spl = result.direct_spl if result.direct_spl is not None else result.spl
        if result.direct_spl is None:
            typer.secho(
                "output-mode=element but no direct SPL available — using combined SPL.",
                fg=typer.colors.YELLOW,
            )
    else:
        primary_spl = result.spl

    responses = {
        "Horn SPL (dB)": primary_spl,
        "Impedance (Ohms)": np.abs(result.impedance),
        "Impedance Real (Ohms)": result.impedance.real,
        "Impedance Imag (Ohms)": result.impedance.imag,
        "Impedance Phase (deg)": (
            result.impedance_phase_deg
            if result.impedance_phase_deg is not None
            else np.angle(result.impedance) * 180.0 / np.pi
        ),
        "Excursion (mm)": result.excursion,
        "Cone Velocity (m/s)": (
            result.cone_velocity
            if result.cone_velocity is not None
            else np.zeros_like(result.excursion)
        ),
        "Phase (degrees)": (
            np.degrees(result.phase)
            if result.phase is not None
            else np.zeros_like(result.spl)
        ),
    }

    # CRIT-3 fix: export acoustic-power-based SPL (dB/W/m Hornresp reference).
    # When driver.sensitivity_db is set in the driver YAML, spl_power_based
    # applies the calibration so pyhorn matches Hornresp at V=2.83.
    # Example YAML: sensitivity_db: -15.0  # dB offset to match Hornresp dB/W/m
    if result.spl_power_based is not None:
        responses["SPL dB/W/m (sensitivity-calibrated)"] = result.spl_power_based
    if result.acoustic_power is not None:
        responses["Acoustic Power (W)"] = result.acoustic_power

    # Add group delay column in the mode selected by --filter-delay-mode
    if filter_delay_mode == "per_period":
        gd_label = "Group delay per period (dimensionless)"
        gd_vals = (
            result.group_delay_per_period
            if result.group_delay_per_period is not None
            else np.zeros_like(result.spl)
        )
    else:
        gd_label = "Group delay (ms)"
        gd_vals = (
            result.group_delay
            if result.group_delay is not None
            else np.zeros_like(result.spl)
        )
    responses[gd_label] = gd_vals

    # ── Apply filter bands from YAML (post-processing) ─────────────────────────
    _filter_yaml_path: Optional[Path] = filter_path or filter_yaml
    if _filter_yaml_path is not None:
        from pyhorn_core.solver.filter_schematic import FilterBand
        from pyhorn_ui.server import _apply_filter_bands

        try:
            with open(_filter_yaml_path) as fh:
                filter_data = yaml.safe_load(fh)
            band_list = (
                filter_data.get("filter_bands", [])
                if isinstance(filter_data, dict)
                else []
            )
            bands: list[FilterBand] = [
                FilterBand(
                    type=b.get("type", "peakingEQ"),
                    frequency=float(b.get("frequency", 1000)),
                    q=float(b.get("q", 1.0)),
                    gain_db=float(b.get("gain_db", 0.0)),
                    order=int(b.get("order", 2)),
                    enabled=b.get("enabled", True),
                )
                for b in band_list
            ]
        except Exception as exc:
            typer.secho(f"Error reading filter YAML: {exc}", fg=typer.colors.RED)
            raise typer.Exit(code=1)

        if bands:
            filt_spl, filt_imp, filt_phase, filt_mag_db = _apply_filter_bands(
                freqs,
                np.array(primary_spl),
                np.abs(result.impedance),
                (
                    np.degrees(result.phase)
                    if result.phase is not None
                    else np.zeros_like(result.spl)
                ),
                bands,
            )
            responses["Filtered SPL (dB)"] = np.array(filt_spl)
            responses["Filtered Impedance (Ohms)"] = np.array(filt_imp)
            responses["Filtered Phase (degrees)"] = np.array(filt_phase)
            responses["Filter contribution (dB)"] = np.array(filt_mag_db)
            typer.echo(
                f"Applied {len([b for b in bands if b.enabled])} filter band(s) from {_filter_yaml_path}"
            )

    # Add second tone distortion column when available (single-segment horns only)
    if result.second_tone_distortion is not None:
        responses["Second tone distortion (dB rel)"] = result.second_tone_distortion

    # Add thermal power compression column when --voice-coil-temp was provided
    if thermal_compression and result.thermal_compression_db is not None:
        responses["Thermal compression (dB)"] = result.thermal_compression_db

    # Flag TMM numerical artifacts in the export
    if result.numerical_artifacts:
        artifact_flags = np.zeros_like(result.spl)
        for afreq in result.numerical_artifacts:
            idx = np.argmin(np.abs(freqs - afreq))
            artifact_flags[idx] = 1.0
        responses["TMM artifact flag"] = artifact_flags

    # Add separate SPL decomposition columns when available (BLH with direct radiator path)
    if result.horn_spl is not None:
        responses["Horn component SPL (dB)"] = result.horn_spl
    if result.direct_spl is not None:
        responses["Direct radiator SPL (dB)"] = result.direct_spl

    # Add off-axis SPL columns when --off-axis-angles is specified
    if off_axis_spl is not None:
        for angle_str, rel_spl in off_axis_spl.items():
            responses[f"Off-axis {angle_str}° SPL (dB rel)"] = rel_spl

    # Add system efficiency (electrical → acoustic conversion)
    if result.efficiency_pct is not None:
        responses["Efficiency (%)"] = result.efficiency_pct

    # Add driver electrical input power — Hornresp page 105
    if result.electrical_input_power is not None:
        responses["Driver Power (W)"] = result.electrical_input_power

    # Add diaphragm pressure — Hornresp pages 124-125 (Pa, complex; only for BLH)
    if result.diaphragm_pressure_total is not None:
        responses["Diaphragm Pressure Total (Pa)"] = np.abs(
            result.diaphragm_pressure_total
        )
    if result.diaphragm_pressure_horn_side is not None:
        responses["Diaphragm Pressure Horn Side (Pa)"] = np.abs(
            result.diaphragm_pressure_horn_side
        )
    if result.diaphragm_pressure_direct_side is not None:
        responses["Diaphragm Pressure Direct Side (Pa)"] = np.abs(
            result.diaphragm_pressure_direct_side
        )

    # Add particle velocity at throat, mouth, port — Hornresp page 106 (m/s peak)
    if result.particle_velocity_throat is not None:
        responses["Particle Velocity Throat (m/s)"] = result.particle_velocity_throat
    if result.particle_velocity_mouth is not None:
        responses["Particle Velocity Mouth (m/s)"] = result.particle_velocity_mouth
    if result.particle_velocity_port is not None:
        responses["Particle Velocity Port (m/s)"] = result.particle_velocity_port

    # Add throat acoustic impedance (complex) — acoustic impedance at horn throat
    if result.throat_impedance is not None:
        responses["Throat Impedance (Ohms)"] = np.abs(result.throat_impedance)
        responses["Throat Impedance Real (Ohms)"] = result.throat_impedance.real
        responses["Throat Impedance Imag (Ohms)"] = result.throat_impedance.imag

    # Add notched SPL when --notch-filter was applied
    if result.spl_notched is not None:
        responses["SPL with notch filter (dB)"] = result.spl_notched

    # Add radiation angle (mean -6 dB beamwidth half-angle) when computed
    if result.radiation_angle is not None:
        responses["Radiation Angle (deg)"] = np.full_like(
            result.spl, result.radiation_angle
        )

    # Add Direction Index (DI) per angle — Hornresp page 94
    if result.direction_index is not None and result.off_axis_angles is not None:
        for j, ang in enumerate(result.off_axis_angles):
            responses[f"DI {ang:.0f}° (dB)"] = result.direction_index[:, j]

    # 6. Generate Outputs
    horn_output_dir = output_dir / horn_name
    horn_output_dir.mkdir(parents=True, exist_ok=True)

    if export_csv:
        csv_path = horn_output_dir / "response.csv"
        export_to_csv(
            freqs,
            responses,
            csv_path,
            futtrup_gdlimit_ms=result.futtrup_gdlimit if futtrup_gd else None,
        )
        typer.echo(f"Exported CSV to {csv_path}")

    if export_json:
        json_path = horn_output_dir / "response.json"
        export_to_json(
            freqs,
            responses,
            json_path,
            futtrup_gdlimit_ms=result.futtrup_gdlimit if futtrup_gd else None,
        )
        typer.echo(f"Exported JSON to {json_path}")

    if export_wav is not None:
        _phase = result.phase if result.phase is not None else np.zeros_like(result.spl)
        export_impulse_to_wav(freqs, result.spl, _phase, export_wav)
        typer.echo(f"Exported impulse response WAV to {export_wav}")

    if export_frd is not None:
        _phase_rad = (
            result.phase if result.phase is not None else np.zeros_like(result.spl)
        )
        _phase_deg = np.degrees(_phase_rad)
        export_to_frd(freqs, primary_spl, _phase_deg, export_frd)
        typer.echo(f"Exported FRD to {export_frd}")

    # Warn about detected TMM numerical artifacts (~1847 Hz region is known)
    if result.numerical_artifacts:
        artifact_str = ", ".join(f"{f:.0f} Hz" for f in result.numerical_artifacts)
        typer.secho(
            f"⚠️  TMM numerical artifacts detected at: {artifact_str}  "
            f"These isolated SPL spikes/dips are calculation artifacts, not physical response. "
            f"Ignore them when interpreting the response curve.",
            fg=typer.colors.YELLOW,
        )

    if plot:
        plot_path = horn_output_dir / "response_plot.png"

        # Determine target SPL
        ref_spl = target_spl if target_spl is not None else driver.reference_spl
        ref_impedance = target_impedance if target_impedance is not None else driver.re
        ref_excursion = None
        excursion_target_label = None
        if target_excursion is not None:
            ref_excursion = target_excursion
            excursion_target_label = f"Target ({target_excursion:.2f} mm)"
        elif driver.xmax > 0:
            ref_excursion = driver.xmax * 1000.0
            excursion_target_label = f"Xmax ({ref_excursion:.2f} mm)"

        plot_simulation_results(
            result,
            plot_path,
            title=f"Horn Acoustic Response",
            target_spl=ref_spl,
            target_impedance=ref_impedance,
            target_excursion=ref_excursion,
            target_excursion_label=excursion_target_label,
            output_mode=output_mode,
            plot_phase=plot_phase,
            plot_distortion=plot_distortion,
            polar_freq=polar_freq if polar_freq and polar_freq > 0 else None,
            show_spectrogram=spectrogram,
            spectrogram_window_ms=spectrogram_window_ms,
            spectrogram_overlap=spectrogram_overlap,
            plot_spl_only=spl_only,
        )
        typer.echo(f"Generated plot at {plot_path}")

    # Time-domain analysis (impulse, step, CSD waterfall)
    if plot and result.pressure is not None:
        # IFFT requires uniform frequency spacing — interpolate from logspace grid
        df_uniform = (freqs[-1] - freqs[0]) / (n_points - 1)
        freqs_uniform = np.arange(freqs[0], freqs[-1] + df_uniform / 2, df_uniform)
        pressure_uniform = np.interp(
            freqs_uniform, freqs, np.abs(result.pressure)
        ) * np.exp(
            1j * np.interp(freqs_uniform, freqs, np.unwrap(np.angle(result.pressure)))
        )

        td = compute_csd(freqs_uniform, pressure_uniform)

        waterfall_path = horn_output_dir / "csd_waterfall.png"
        plot_waterfall(
            td.csd_freqs,
            td.csd_times_ms,
            td.csd_db,
            waterfall_path,
            title=f"{horn_name} — CSD Waterfall",
        )
        typer.echo(f"Generated CSD waterfall at {waterfall_path}")

        impulse_path = horn_output_dir / "impulse_step.png"
        plot_impulse_step(
            td.time_ms,
            td.impulse,
            td.step,
            impulse_path,
            title=f"{horn_name} — Impulse & Step Response",
        )
        typer.echo(f"Generated impulse/step plot at {impulse_path}")

    # Standalone spectrogram PNG (--spectrogram flag)
    if spectrogram and result.pressure is not None:
        spec_path = horn_output_dir / "spectrogram.png"
        try:
            time_ms, freq_bins, stft_db = compute_spectrogram(
                result.freqs,
                result.pressure,
                window_ms=spectrogram_window_ms,
                overlap=spectrogram_overlap,
            )
            fig, _ = plot_spectrogram(
                time_ms,
                freq_bins,
                stft_db,
                f_min=fmin,
                f_max=fmax,
                title=f"{horn_name} — Spectrogram (window={spectrogram_window_ms:.0f} ms, overlap={spectrogram_overlap:.0%})",
            )
            fig.savefig(str(spec_path))
            typer.echo(f"Generated spectrogram at {spec_path}")
        except Exception as e:
            typer.secho(f"Spectrogram generation failed: {e}", fg=typer.colors.YELLOW)

    # 7. Acoustic Characteristics Report
    segments = result.segments
    if segments:
        total_len = sum(s[0] for s in segments)
        # Boundary areas: prefer the user-specified throat/mouth on the geometry.
        # segments[0][1] / segments[-1][1] are segment-AVERAGED areas — for a
        # monotonic profile they sit slightly off the true endpoints.
        throat_area = horn.throat_area if horn.throat_area > 0 else segments[0][1]
        mouth_area = horn.mouth_area if horn.mouth_area > 0 else segments[-1][1]

        # Approximate internal volume in Liters
        volume_m3 = sum(
            s[0] * (s[1] + segments[max(0, i - 1)][1]) / 2
            for i, s in enumerate(segments)
        )
        volume_l = volume_m3 * 1000.0

        # Equivalent Cutoff Frequency (Exponential Flare Rate)
        # Correctly handles hyperbolic profiles using the Le Cléac'h transcendental equation
        # (matches the UI's HornMetrics fc calculation)
        if total_len > 0 and throat_area > 0:
            profile_type = getattr(horn, "profile_type", None) or "exponential"
            hyperbolic_t = getattr(horn, "hyperbolic_t", None) or 1.0
            expansion = mouth_area / throat_area
            pt_lower = profile_type.lower()

            if pt_lower == "hyperbolic":
                u = _solve_hyperbolic_u(math.sqrt(expansion), hyperbolic_t)
                m = u / total_len
                fc = (m * C) / (2 * np.pi)
            elif pt_lower in ("exponential", "parabolic"):
                m = np.log(expansion) / total_len
                fc = (m * C) / (4 * np.pi)
            else:
                # Conical or unknown profile — no exponential flare, fc is effectively 0
                m = 0.0
                fc = 0.0
            fq = C / (4 * total_len)
        else:
            fc = 0.0
            fq = 0.0

        geometry = horn.geometry_diagnostics()
        geometry_lines = []
        if geometry:
            geometry_lines.append("\nGeometry Diagnostics:\n")
            if "segment_count" in geometry:
                segment_label = (
                    "Rectangular Segments"
                    if horn.rectangular_segments
                    else "Conical Segments"
                )
                geometry_lines.append(
                    f"  {segment_label}:   {int(geometry['segment_count'])}\n"
                )
                geometry_lines.append(
                    f"  Segment Lengths:    {geometry['min_segment_length_m']:.3f}-{geometry['max_segment_length_m']:.3f} m\n"
                )
                geometry_lines.append(
                    f"  Area Range:         {geometry['min_area_m2'] * 10000:.1f}-{geometry['max_area_m2'] * 10000:.1f} cm²\n"
                )
                geometry_lines.append(
                    f"  Max Area Step:      {geometry['max_area_step_ratio']:.3f}:1\n"
                )
            if "min_width_m" in geometry:
                geometry_lines.append(
                    f"  Width Range:        {geometry['min_width_m']:.3f}-{geometry['max_width_m']:.3f} m\n"
                )
                geometry_lines.append(
                    f"  Height Range:       {geometry['min_height_m']:.3f}-{geometry['max_height_m']:.3f} m\n"
                )
            if "max_bend_angle_deg" in geometry:
                geometry_lines.append(
                    f"  Sharpest Bend:      {geometry['max_bend_angle_deg']:.1f} deg\n"
                )
                geometry_lines.append(
                    f"  Mean Bend:          {geometry['mean_bend_angle_deg']:.1f} deg\n"
                )
            if horn.lem_step_model and horn.lem_step_model.lower() not in {
                "ideal",
                "none",
            }:
                geometry_lines.append(f"  LEM Step Model:     {horn.lem_step_model}\n")
                geometry_lines.append(
                    f"  LEM Strength:       {horn.lem_step_strength:.2f}\n"
                )
                if horn.lem_step_resistance > 0:
                    geometry_lines.append(
                        f"  LEM Resistance:     {horn.lem_step_resistance:.3f}\n"
                    )

        report = "".join(
            [
                "\n--- Acoustic Characteristics Report ---\n",
                f"Total Path Length:  {total_len:.3f} m\n",
                f"Throat Area:        {throat_area * 10000:.1f} cm²\n",
                f"Mouth Area:         {mouth_area * 10000:.1f} cm²\n",
                f"Internal Volume:    {volume_l:.2f} Liters\n",
                f"Flare Cutoff (fc):  {fc:.1f} Hz\n",
                f"1/4 Wave Tuning:    {fq:.1f} Hz\n",
                *geometry_lines,
                "---------------------------------------\n",
            ]
        )

        typer.echo(report)
        report_path = horn_output_dir / "report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        typer.echo(f"Saved report to {report_path}")

    # ── Thermal Power Compression Report ──────────────────────────────────────
    if thermal_compression and result.thermal_compression_db is not None and voice_coil_temp is not None:
        tcdb = result.thermal_compression_db
        alpha = getattr(driver, "alpha_re", 0.00393)
        re_hot = driver.re * (1.0 + alpha * (voice_coil_temp - 20.0))
        typer.echo(
            f"\n--- Thermal Power Compression Report (T={voice_coil_temp:.0f}°C) ---\n"
            f"  Re nominal (20°C):  {driver.re:.3f} Ω\n"
            f"  Re heated ({voice_coil_temp:.0f}°C):   {re_hot:.3f} Ω\n"
            f"  Resistance increase: +{re_hot - driver.re:.3f} Ω "
            f"({(re_hot / driver.re - 1) * 100:.1f}%)\n"
            f"  SPL compression:     avg={np.mean(tcdb):.2f} dB, "
            f"max={np.min(tcdb):.2f} dB\n"
            f"  Least compressed:    {freqs[np.argmax(tcdb)]:.1f} Hz "
            f"({np.max(tcdb):.2f} dB)\n"
            f"  Most compressed:     {freqs[np.argmin(tcdb)]:.1f} Hz "
            f"({np.min(tcdb):.2f} dB)\n"
        )

    if radiation_summary and horn.mouth_area and horn.mouth_area > 0:
        _print_radiation_summary(freqs, primary_spl, horn.mouth_area)

    if plot_3d:
        if horn.coordinates:
            from pyhorn_core.output.plotter import plot_horn_2d_folded

            plot_2d_path = horn_output_dir / "horn_2d_folded.png"
            folded_segments = horn.folded_plot_segments()
            edims = horn.enclosure_dims if horn.enclosure_dims else (0.5, 0.5)
            project_has_chambers = project_config is not None
            wall_t = getattr(project, "thickness", 0.0) if project_has_chambers else 0.0
            rc_data = (
                getattr(project, "rear_chamber", None) if project_has_chambers else None
            )
            rc_tuple = None
            if rc_data is not None:
                rc_tuple = (rc_data.width, rc_data.height, rc_data.depth)
            if folded_segments is not None:
                plot_horn_2d_folded(
                    folded_segments,
                    horn.coordinates,
                    edims,
                    plot_2d_path,
                    horn.driver_coord,
                    throat_chamber_side=_folded_throat_chamber_side(horn),
                    wall_t=wall_t,
                    rear_chamber=rc_tuple,
                )
                typer.echo(f"Generated 2D folded schematic at {plot_2d_path}")

        from pyhorn_core.output.plotter import plot_horn_3d

        plot_3d_path = horn_output_dir / "horn_3d.png"
        if segments is not None:
            plot_horn_3d(
                segments,
                plot_3d_path,
                width=horn.width,
                width_profile=result.segment_widths,
            )
            typer.echo(f"Generated 3D straightened schematic at {plot_3d_path}")

    typer.secho("Calculation complete!", fg=typer.colors.GREEN)


def compare(
    horns: list[Path] = typer.Argument(
        ..., help="List of horn YAML configs to compare"
    ),
    driver_config: Path = typer.Option(
        ..., "--driver", "-d", help="Path to driver config"
    ),
    output_dir: Path = typer.Option(
        Path("./outputs/comparison"),
        "--output-dir",
        "-o",
        help="Directory to save outputs",
    ),
    target_spl: Optional[float] = typer.Option(
        None, help="Target SPL for reference line (defaults to driver reference SPL)"
    ),
):
    """
    Compare the SPL response of multiple horn designs on a single plot.
    """
    import matplotlib.pyplot as plt
    from pyhorn_core.solver.models import horn_response

    driver = parse_driver_specs(driver_config)
    freqs = np.logspace(np.log10(20), np.log10(5000), 500)

    fig, ax = plt.subplots(figsize=(9, 5))
    output_dir.mkdir(parents=True, exist_ok=True)

    for horn_path in horns:
        try:
            horn = parse_horn_geometry(horn_path)
            res = horn_response(freqs, driver, horn)
            ax.semilogx(freqs, res.spl, label=horn_path.stem, linewidth=0.8)
            typer.echo(f"Simulated {horn_path.stem}")
        except Exception as e:
            typer.secho(f"Failed to simulate {horn_path}: {e}", fg=typer.colors.RED)

    ref_spl = target_spl if target_spl is not None else driver.reference_spl
    ax.axhline(
        ref_spl,
        color="#f59e0b",
        linestyle="--",
        linewidth=0.5,
        label=f"Target ({ref_spl:.1f} dB)",
    )

    from pyhorn_core.solver.models import infinite_baffle_response

    ib_spl = infinite_baffle_response(freqs, driver)
    ax.semilogx(
        freqs,
        ib_spl,
        color="#9ca3af",
        linestyle="--",
        linewidth=0.6,
        label="Infinite Baffle",
    )

    ax.set_xlim(20, 5000)
    from pyhorn_core.output.plotter import _apply_style

    _apply_style(ax, xlabel="Frequency (Hz)", ylabel="SPL (dB @ 1W/1m)", freq_axis=True)
    ax.set_title("Horn SPL Comparison", fontsize=10, fontweight="medium")
    ax.legend(fontsize=7, framealpha=0.6, edgecolor="none")

    out_file = output_dir / "spl_compare.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()

    typer.secho(f"Comparison plot saved to {out_file}", fg=typer.colors.GREEN)


def derive_ts(
    fs: float = typer.Option(..., help="Resonant frequency (Hz)"),
    qes: float = typer.Option(..., help="Electrical Q"),
    qms: float = typer.Option(..., help="Mechanical Q"),
    vas: float = typer.Option(..., help="Equivalent volume (L)"),
    re: float = typer.Option(..., help="DC Resistance (Ohms)"),
    sd: float = typer.Option(..., help="Piston area (cm²)"),
    output_yaml: bool = typer.Option(
        False,
        "--output-yaml",
        help="Output as YAML snippet ready to paste into a driver YAML file.",
    ),
):
    """
    Derive mechanical parameters from basic Thiele-Small measurements.
    Outputs values in SI units ready for use in Pyhorn configurations.
    """
    # ── Input validation ──────────────────────────────────────────────────────
    for name, val in [
        ("fs", fs),
        ("Qes", qes),
        ("Qms", qms),
        ("Vas", vas),
        ("Re", re),
        ("Sd", sd),
    ]:
        if val <= 0:
            raise typer.BadParameter(f"{name} must be positive, got {val}")
    # ── Convert inputs to SI ──────────────────────────────────────────────────
    vas_m3 = vas / 1000.0
    sd_m2 = sd / 10000.0

    # Derived
    qts = (qes * qms) / (qes + qms)
    cas = vas_m3 / (RHO * C**2)
    cms = cas / (sd_m2**2)
    mms = 1.0 / ((2 * np.pi * fs) ** 2 * cms)
    rms = (2 * np.pi * fs * mms) / qms
    bl = np.sqrt((2 * np.pi * fs * mms * re) / qes)

    if output_yaml:
        # Convert all numpy scalars to native Python floats for clean YAML output
        bl_val = float(bl)
        data = {
            "sd": float(sd_m2),
            "re": float(re),
            "bl": float(bl_val),
            "cms": float(cms),
            "mms": float(mms),
            "rms": float(rms),
            "qts": float(qts),
            "qes": float(qes),
            "qms": float(qms),
            "vas": float(vas_m3),
            "fs": float(fs),
        }
        typer.echo("# Derived T-S parameters (SI units)")
        typer.echo(yaml.dump(data, default_flow_style=False, sort_keys=False).rstrip())
    else:
        typer.echo(f"Derived T-S Parameters (SI Units):")
        typer.echo(f"  fs:  {fs:.2f} Hz")
        typer.echo(f"  qts: {qts:.3f}")
        typer.echo(f"  qes: {qes:.3f}")
        typer.echo(f"  qms: {qms:.3f}")
        typer.echo(f"  vas: {vas_m3:.5f} m³")
        typer.echo(f"  re:  {re:.2f} Ohms")
        typer.echo(f"  bl:  {bl:.2f} N/A")
        typer.echo(f"  mms: {mms:.5f} kg")
        typer.echo(f"  cms: {cms:.5e} m/N")
        typer.echo(f"  rms: {rms:.3f} kg/s")
        typer.echo(f"  sd:  {sd_m2:.5f} m²")
