"""Optimization commands: optimize, fold-optimized."""

from pathlib import Path
from typing import Optional

import numpy as np
import typer
import yaml

from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry
from pyhorn_core.solver.models import horn_response, C
try:
    from pyhorn_fold import extrapolate_folded_horn
    _HAS_PYHORN_FOLD = True
except ImportError:
    extrapolate_folded_horn = None
    _HAS_PYHORN_FOLD = False
from pyhorn_core.output.plotter import plot_horn_2d_folded
from pyhorn_core.solver.optimizer import (
    OptimizationConfig,
    optimize as run_optimize,
)

from ._shared import _folded_throat_chamber_side


def _write_optimizer_yaml(path, result, driver_name, fmin, fmax):
    """Write an optimised horn geometry as a YAML file usable by `pyhorn calculate`."""
    p = result.params
    path.write_text(
        f"# Optimized horn geometry - {result.profile_type} profile\n"
        f"# Driver: {driver_name}, Band: {fmin:.0f}-{fmax:.0f} Hz\n"
        f"# Cost: {result.cost:.3f} (ripple={result.flatness_db:.1f} dB, "
        f"SPL={result.mean_spl:.1f} dB)\n"
        f"enclosure_type: {result.horn.enclosure_type}\n"
        f"profile_type: {result.profile_type}\n"
        f"throat_area: {p['throat_area']:.6f}\n"
        f"mouth_area: {p['mouth_area']:.6f}\n"
        f"path_length: {p['path_length']:.4f}\n"
        f"n_segments: 100\n"
        f"lrc: {p['lrc']:.4f}\n"
        f"vrc: {p['lrc'] * p['throat_area']:.6f}\n"
        f"vtc: {p['vtc']:.6f}\n"
    )


def _write_folded_optimizer_yaml(path, result, horn, driver_name, fmin, fmax):
    """Write a folded version of an optimized horn geometry to YAML."""
    _write_folded_horn_yaml(
        path,
        horn,
        header_lines=[
            f"# Folded layout extrapolated from optimized {result.profile_type} profile",
            f"# Driver: {driver_name}, Band: {fmin:.0f}-{fmax:.0f} Hz",
            f"# Cost: {result.cost:.3f} (ripple={result.flatness_db:.1f} dB, SPL={result.mean_spl:.1f} dB)",
        ],
    )


def _build_folded_layout(horn):
    """Build a folded_layout dict with panels from rectangular_segments.

    Each panel contains:
      x1, y1, x2, y2 — segment endpoints (m)
      width           — panel width = segment edge length (m)
      height          — panel thickness = enclosure dim perpendicular to panel (m)
      angle           — angle of segment edge vs x-axis (radians)
      connection      — "throat" for first panel, "panel_N" for subsequent
    """
    import math

    panels = []
    depth, height = horn.enclosure_dims if horn.enclosure_dims else (0.0, 0.0)
    segs = horn.rectangular_segments or []
    for i, seg in enumerate(segs):
        x1, y1, x2, y2, length = seg
        dx = x2 - x1
        dy = y2 - y1
        edge_length = math.sqrt(dx * dx + dy * dy) or length
        # Determine if panel is vertical (dx ≈ 0) or horizontal (dy ≈ 0)
        if abs(dx) < 1e-9:  # vertical panel
            panel_height = depth
            panel_width = edge_length
        else:  # horizontal panel
            panel_height = edge_length
            panel_width = height
        angle = math.atan2(dy, dx)
        connection = "throat" if i == 0 else f"panel_{i - 1}"
        panels.append({
            "x1": round(x1, 6),
            "y1": round(y1, 6),
            "x2": round(x2, 6),
            "y2": round(y2, 6),
            "width": round(panel_width, 6),
            "height": round(panel_height, 6),
            "angle": round(angle, 6),
            "connection": connection,
        })
    return panels


def _write_folded_horn_yaml(path, horn, header_lines):
    """Write a folded horn geometry YAML with optional header comments."""
    data = {
        "enclosure_type": horn.enclosure_type,
        "throat_area": round(horn.throat_area, 6),
        "mouth_area": round(horn.mouth_area, 6),
        "path_length": round(horn.path_length, 4),
        "width": round(horn.width or 0.0, 6),
        "lrc": round(horn.lrc, 4),
        "vrc": round(horn.vrc, 6),
        "vtc": round(horn.vtc, 6),
        "enclosure_dims": [round(value, 4) for value in horn.enclosure_dims or ()],
        "driver_coord": [round(value, 4) for value in horn.driver_coord or ()],
        "coordinates": [
            [round(coord[0], 4), round(coord[1], 4)] for coord in horn.coordinates or []
        ],
        "conical_segments": [
            [round(seg[0], 6), round(seg[1], 6), round(seg[2], 4)]
            for seg in horn.conical_segments or []
        ],
        "rectangular_segments": [
            [
                round(seg[0], 6),
                round(seg[1], 6),
                round(seg[2], 6),
                round(seg[3], 6),
                round(seg[4], 4),
            ]
            for seg in horn.rectangular_segments or []
        ],
    }
    folded_layout = _build_folded_layout(horn)
    if folded_layout:
        data["folded_layout"] = {"panels": folded_layout}
    header = "".join(f"{line}\n" for line in header_lines)
    path.write_text(header + yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _plot_optimizer_results(results, driver, config, output_dir):
    """SPL comparison plot of top optimizer results + infinite baffle reference."""
    import matplotlib.pyplot as plt
    from pyhorn_core.solver.models import infinite_baffle_response
    from pyhorn_core.output.plotter import _apply_style

    freqs = np.logspace(np.log10(20), np.log10(config.fmax), 500)
    fig, ax = plt.subplots(figsize=(9, 5))

    for r in results:
        sim = horn_response(freqs, driver, r.horn)
        ax.semilogx(
            freqs,
            sim.spl,
            linewidth=0.8,
            label=f"{r.profile_type} (ripple {r.flatness_db:.1f} dB, SPL {r.mean_spl:.1f} dB)",
        )

    ax.semilogx(
        freqs,
        infinite_baffle_response(freqs, driver),
        color="#9ca3af",
        linestyle="--",
        linewidth=0.6,
        label="Infinite Baffle",
    )
    ax.axhline(
        driver.reference_spl,
        color="#f59e0b",
        linestyle="--",
        linewidth=0.5,
        label=f"Ref ({driver.reference_spl:.1f} dB)",
    )

    ax.set_xlim(20, config.fmax)
    _apply_style(ax, xlabel="Frequency (Hz)", ylabel="SPL (dB @ 1W/1m)", freq_axis=True)
    ax.set_title("Optimized Horn Comparison", fontsize=10, fontweight="medium")
    ax.legend(fontsize=7, framealpha=0.6, edgecolor="none")
    plt.tight_layout()
    plt.savefig(output_dir / "optimize_compare.png", dpi=180)
    plt.close()


def optimize(
    driver_config: Path = typer.Option(
        ..., "--driver", "-d", help="Path to driver JSON/YAML config"
    ),
    output_dir: Path = typer.Option(
        Path("./outputs/optimize"),
        "--output-dir",
        "-o",
        help="Directory to save results",
    ),
    fmin: float = typer.Option(80.0, help="Target band lower frequency (Hz)"),
    fmax: float = typer.Option(5000.0, help="Target band upper frequency (Hz)"),
    enclosure_type: str = typer.Option(
        "BLH", "--enclosure", help="Enclosure type: FLH or BLH"
    ),
    max_path_length: Optional[float] = typer.Option(
        None, "--max-path-length", help="Maximum horn path length (m)"
    ),
    max_mouth_area: Optional[float] = typer.Option(
        None, "--max-mouth-area", help="Maximum mouth area (m²)"
    ),
    profiles: Optional[str] = typer.Option(
        None, help="Comma-separated profile types (default: all four)"
    ),
    max_iter: int = typer.Option(150, help="Max optimizer iterations per profile type"),
    top_n: int = typer.Option(3, help="Number of top designs to output"),
    seed: Optional[int] = typer.Option(None, help="Random seed for reproducibility"),
    min_expansion_ratio: float = typer.Option(
        4.0,
        help="Minimum mouth/throat area ratio to enforce during optimization",
    ),
    throat_penalty_weight: float = typer.Option(
        0.5,
        help="Penalty weight for throat areas larger than driver Sd",
    ),
    plot: bool = typer.Option(True, help="Generate comparison SPL plot"),
    enclosure_depth: Optional[float] = typer.Option(
        None, help="Folded layout enclosure depth (m)"
    ),
    enclosure_height: Optional[float] = typer.Option(
        None, help="Folded layout enclosure height (m)"
    ),
    driver_x: Optional[float] = typer.Option(
        None, help="Driver x-position inside the folded enclosure (m)"
    ),
    driver_y: Optional[float] = typer.Option(
        None, help="Driver y-position inside the folded enclosure (m)"
    ),
    enclosure_width: Optional[float] = typer.Option(
        None,
        "--enclosure-width",
        "--folded-width",
        help="Internal enclosure width used as horn width for folded export (m).",
    ),
):
    """
    Find optimal horn geometry for a given driver.

    Runs scipy differential_evolution independently for each flare profile,
    searching over throat area, mouth area, path length, rear chamber length,
    and throat chamber volume.  Designs whose cutoff frequency exceeds --fmin
    are rejected.
    """
    fold_values = [enclosure_depth, enclosure_height, driver_x, driver_y]
    if any(value is not None for value in fold_values) and not all(
        value is not None for value in fold_values
    ):
        typer.secho(
            "Folded export requires --enclosure-depth, --enclosure-height, --driver-x, and --driver-y together.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)
    if any(value is not None for value in fold_values) and enclosure_width is None:
        typer.secho(
            "Folded export requires --enclosure-width.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(code=1)

    driver = parse_driver_specs(driver_config)

    config = OptimizationConfig(
        fmin=fmin,
        fmax=fmax,
        enclosure_type=enclosure_type.upper(),
        max_iter=max_iter,
        top_n=top_n,
        seed=seed,
        min_expansion_ratio=min_expansion_ratio,
        throat_area_penalty_weight=throat_penalty_weight,
    )
    if max_path_length is not None:
        config.path_length_range = (config.path_length_range[0], max_path_length)
    if max_mouth_area is not None:
        config.mouth_area_range = (config.mouth_area_range[0], max_mouth_area)
    if profiles is not None:
        config.profile_types = [p.strip() for p in profiles.split(",")]

    typer.echo(f"Optimizing horn for {driver_config.stem}")
    typer.echo(f"  Band: {fmin}-{fmax} Hz, Enclosure: {config.enclosure_type}")
    typer.echo(f"  Profiles: {', '.join(config.profiles)}")
    typer.echo(f"  Min expansion ratio: {config.min_expansion_ratio:.2f}:1")
    typer.echo(f"  Throat penalty weight: {config.throat_area_penalty_weight:.2f}")
    typer.echo(f"  Max iterations per profile: {max_iter}")
    typer.echo()

    results = run_optimize(driver, config, progress_callback=typer.echo)

    if not results:
        typer.secho(
            "Optimization failed - no valid designs found.", fg=typer.colors.RED
        )
        raise typer.Exit(code=1)

    # ── Summary table ────────────────────────────────────────────────────
    typer.echo()
    typer.echo("Results (sorted by cost, lower = better):")
    typer.echo(
        f"{'Rank':<5} {'Profile':<14} {'Cost':<8} {'SPL':<8} "
        f"{'Ripple':<9} {'Bass -':<8} {'Exc OK':<7} {'Evals'}"
    )
    typer.echo("-" * 72)
    for i, r in enumerate(results):
        typer.echo(
            f"{i+1:<5} {r.profile_type:<14} {r.cost:<8.3f} {r.mean_spl:<8.1f} "
            f"{r.flatness_db:<9.1f} {r.bass_deficit_db:<8.1f} "
            f"{'yes' if r.excursion_ok else 'NO':<7} {r.n_evaluations}"
        )

    # ── Save top-N YAMLs ─────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(results[: config.top_n]):
        yaml_path = output_dir / f"optimized_{i+1}_{r.profile_type}.yaml"
        _write_optimizer_yaml(yaml_path, r, driver_config.stem, fmin, fmax)
        typer.echo(f"Saved {yaml_path}")

    if all(value is not None for value in fold_values):
        assert enclosure_depth is not None
        assert enclosure_height is not None
        assert driver_x is not None
        assert driver_y is not None
        assert enclosure_width is not None
        enclosure_dims = (float(enclosure_depth), float(enclosure_height))
        driver_coord = (float(driver_x), float(driver_y))

        if not _HAS_PYHORN_FOLD:
            raise typer.Exit(
                code=1,
                message="pyhorn_fold is required for optimization. "
                        "Install with: pip install pyhorn-fold "
                        "or use: pip install -e ../pyhorn_exp/pyhorn_fold",
            )

        for i, r in enumerate(results[: config.top_n]):
            folded_horn = extrapolate_folded_horn(
                r.horn,
                enclosure_dims,
                driver_coord,
                enclosure_width=float(enclosure_width),
            )
            folded_yaml_path = (
                output_dir / f"optimized_{i+1}_{r.profile_type}_folded.yaml"
            )
            _write_folded_optimizer_yaml(
                folded_yaml_path,
                r,
                folded_horn,
                driver_config.stem,
                fmin,
                fmax,
            )
            typer.echo(f"Saved {folded_yaml_path}")
            typer.echo(f"  folded width: {folded_horn.width:.4f} m")

            folded_plot_path = (
                output_dir / f"optimized_{i+1}_{r.profile_type}_folded.png"
            )
            folded_segments = folded_horn.folded_plot_segments()
            if folded_segments is not None and folded_horn.coordinates is not None:
                plot_horn_2d_folded(
                    folded_segments,
                    folded_horn.coordinates,
                    folded_horn.enclosure_dims,
                    folded_plot_path,
                    folded_horn.driver_coord,
                    throat_chamber_side=_folded_throat_chamber_side(folded_horn),
                    title=f"Optimized {r.profile_type.title()} Folded Layout",
                )
                typer.echo(f"Saved {folded_plot_path}")

    # ── Best-design report ───────────────────────────────────────────────
    best = results[0]
    p = best.params
    m = np.log(p["mouth_area"] / p["throat_area"]) / p["path_length"]
    typer.echo()
    typer.echo(f"Best design: {best.profile_type}")
    typer.echo(f"  Throat area:   {p['throat_area'] * 1e4:.1f} cm2")
    typer.echo(f"  Mouth area:    {p['mouth_area'] * 1e4:.1f} cm2")
    typer.echo(f"  Path length:   {p['path_length']:.3f} m")
    typer.echo(f"  Rear chamber:  {p['lrc'] * 100:.1f} cm")
    typer.echo(f"  Throat vol:    {p['vtc'] * 1e6:.1f} cm3")
    typer.echo(f"  Flare cutoff:  {m * C / (4 * np.pi):.1f} Hz")
    typer.echo(f"  Quarter-wave:  {C / (4 * p['path_length']):.1f} Hz")

    # ── Comparison plot ──────────────────────────────────────────────────
    if plot:
        _plot_optimizer_results(results[: config.top_n], driver, config, output_dir)
        typer.echo(f"Saved comparison plot to {output_dir / 'optimize_compare.png'}")

    typer.secho("Optimization complete!", fg=typer.colors.GREEN)


def fold_optimized(
    optimized_horn: Path = typer.Argument(
        ..., help="Path to optimized horn YAML from `pyhorn optimize`"
    ),
    output_yaml: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Path to save the folded horn YAML (defaults next to input)",
    ),
    enclosure_depth: float = typer.Option(
        ..., help="Folded layout enclosure depth (m)"
    ),
    enclosure_height: float = typer.Option(
        ..., help="Folded layout enclosure height (m)"
    ),
    driver_x: float = typer.Option(
        ..., help="Driver x-position inside the folded enclosure (m)"
    ),
    driver_y: float = typer.Option(
        ..., help="Driver y-position inside the folded enclosure (m)"
    ),
    enclosure_width: float = typer.Option(
        ...,
        "--enclosure-width",
        "--folded-width",
        help="Internal enclosure width used as horn width for folded export (m).",
    ),
    plot: bool = typer.Option(True, help="Generate a folded 2D plot"),
):
    """Create a folded horn layout from an optimized horn YAML."""
    if not _HAS_PYHORN_FOLD:
        raise typer.Exit(
            code=1,
            message="pyhorn_fold is required for fold-optimized. "
                    "Install with: pip install pyhorn-fold "
                    "or use: pip install -e ../pyhorn_exp/pyhorn_fold",
        )
    try:
        horn = parse_horn_geometry(optimized_horn)
        folded_horn = extrapolate_folded_horn(
            horn,
            (float(enclosure_depth), float(enclosure_height)),
            (float(driver_x), float(driver_y)),
            enclosure_width=float(enclosure_width),
        )
    except Exception as e:
        typer.secho(f"Error creating folded horn layout: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    yaml_path = (
        output_yaml
        if output_yaml is not None
        else optimized_horn.with_name(f"{optimized_horn.stem}_folded.yaml")
    )
    plot_path = yaml_path.with_suffix(".png")

    _write_folded_horn_yaml(
        yaml_path,
        folded_horn,
        header_lines=[
            "# Folded layout extrapolated from optimized horn geometry",
            f"# Source: {optimized_horn.name}",
        ],
    )
    typer.echo(f"Saved {yaml_path}")
    typer.echo(f"  folded width: {folded_horn.width:.4f} m")

    if plot:
        folded_segments = folded_horn.folded_plot_segments()
        if folded_segments is not None and folded_horn.coordinates is not None:
            plot_horn_2d_folded(
                folded_segments,
                folded_horn.coordinates,
                folded_horn.enclosure_dims,
                plot_path,
                folded_horn.driver_coord,
                throat_chamber_side=_folded_throat_chamber_side(folded_horn),
                title=f"Folded Layout - {optimized_horn.stem}",
            )
            typer.echo(f"Saved {plot_path}")

    typer.secho("Folded horn export complete!", fg=typer.colors.GREEN)
