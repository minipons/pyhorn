"""Optimization commands."""

from pathlib import Path
from typing import Optional

import numpy as np
import typer

from pyhorn_core.config.parser import parse_driver_specs
from pyhorn_core.solver.models import horn_response, C
from pyhorn_core.solver.optimizer import (
    OptimizationConfig,
    optimize as run_optimize,
)


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
):
    """
    Find optimal horn geometry for a given driver.

    Runs scipy differential_evolution independently for each flare profile,
    searching over throat area, mouth area, path length, rear chamber length,
    and throat chamber volume.  Designs whose cutoff frequency exceeds --fmin
    are rejected.
    """
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
