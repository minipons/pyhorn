"""pyhorn CLI entry point — assembles all command groups."""

import math
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

import typer

try:
    __version__ = version("pyhorn")
except PackageNotFoundError:
    try:
        __version__ = version("pyhorn_cli")
    except PackageNotFoundError:
        __version__ = "0.1.0"

from pyhorn_cli.cli.core_commands import calculate, compare, derive_ts
from pyhorn_cli.cli.design_wizard import (
    resize_wizard,
    hornresp,
    chamber_wizard,
    segment_wizard,
    synthesis_wizard,
)
from pyhorn_cli.cli.horn_commands import (
    tapped_horn,
    throat_adapter,
    auto_segment,
    diagnose_spl,
    driver_front_volume,
)
from pyhorn_cli.cli.optimize_commands import optimize
from pyhorn_core.solver.horn_segment import compute_horn_segment

# ─── Standalone top-level commands ───────────────────────────────────────────


def _segment_wizard_impl(
    s1: Optional[float],
    s2: Optional[float],
    l12: Optional[float],
    f12: Optional[float],
):
    s1_m2 = (s1 * 1e-4) if s1 is not None else None
    s2_m2 = (s2 * 1e-4) if s2 is not None else None
    l12_m = (l12 * 0.01) if l12 is not None else None

    try:
        result = compute_horn_segment(s1_m2=s1_m2, s2_m2=s2_m2, l12_m=l12_m, f12_hz=f12)
    except ValueError as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

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

    typer.echo(f"\n  {'─'*50}")
    typer.echo(f"  {'Position':>12}  {'Area (cm²)':>12}  {'Diameter (mm)':>14}")
    typer.echo(f"  {'─'*50}")
    for frac, area_cm2 in result.area_profile:
        equiv_diameter_mm = 2.0 * math.sqrt(area_cm2 / math.pi) * 10.0
        label_suffix = (
            " (throat)" if frac == 0.0 else (" (mouth)" if frac == 1.0 else "")
        )
        typer.echo(
            f"  {frac:.2f}{label_suffix:>14}  {area_cm2:>12.2f}  {equiv_diameter_mm:>14.1f}"
        )
    typer.echo(f"  {'─'*50}")
    typer.echo(
        f"\n  System volume: {result.system_volume_l:.3f} L  (horn + 0.1 L throat chamber)"
    )
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


# ─── Main app ─────────────────────────────────────────────────────────────────


def _build_app() -> typer.Typer:
    app = typer.Typer()

    # Simulation commands — all flat, no nesting
    app.command("calculate")(calculate)
    app.command("compare")(compare)
    app.command("derive-ts")(derive_ts)

    # Design wizards
    app.command("resize-wizard")(resize_wizard)
    app.command("hornresp")(hornresp)
    app.command("chamber-wizard")(chamber_wizard)
    app.command("segment-wizard")(segment_wizard)
    app.command("synthesis-wizard")(synthesis_wizard)

    # Horn commands
    app.command("tapped-horn")(tapped_horn)
    app.command("throat-adapter")(throat_adapter)
    app.command("auto-segment")(auto_segment)
    app.command("diagnose-spl")(diagnose_spl)
    app.command("driver-front-volume")(driver_front_volume)

    # Optimisation
    app.command("optimize")(optimize)

    return app


app = _build_app()


def run() -> None:
    app()


if __name__ == "__main__":
    run()
