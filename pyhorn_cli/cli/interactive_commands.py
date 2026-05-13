"""Interactive CLI commands (require display or have special runtime constraints)."""

import os
from pathlib import Path
from typing import Optional

import typer


def wavefront_edit(
    geometry: Path = typer.Option(..., "--geometry", "-g", help="Path to a pyhorn geometry YAML."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output YAML path."),
    title: Optional[str] = typer.Option(None, "--title", help="Window title for the interactive editor."),
    figsize: tuple[float, float] = typer.Option((12, 8), "--figsize", help="Matplotlib figure size as 'width,height' in inches."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Load geometry, print summary, and exit without opening the editor."),
    simulate: bool = typer.Option(False, "--simulate", help="After editing, run 2-D wave solver and save pressure-field PNG."),
    frequency: float = typer.Option(1000.0, "--frequency", "-f", help="Drive frequency in Hz (used with --simulate)."),
    grid_size: int = typer.Option(200, "--grid-size", help="Square grid cells per side for --simulate (higher = slower)."),
):
    """Interactive horn-wall vertex editor using matplotlib."""
    display = os.environ.get("DISPLAY", "")
    is_windows = os.name == "nt"
    if not is_windows and not display and not dry_run:
        typer.secho(
            "No DISPLAY found. Interactive editing requires a windowing system.\n"
            "Use --dry-run to verify geometry without opening the editor.",
            fg=typer.colors.YELLOW,
        )
        raise typer.Exit(code=1)

    if isinstance(figsize, str):
        try:
            w, h = [float(x.strip()) for x in figsize.split(",")]
            figsize = (w, h)
        except Exception:
            typer.secho(f"Invalid --figsize format '{figsize}'. Use 'width,height' e.g. '12,8'.", fg=typer.colors.RED)
            raise typer.Exit(code=1)

    from pyhorn_core.solver.wavefront import edit_horn_geometry_from_yaml

    typer.echo(f"Loading geometry from {geometry} ...")

    if dry_run:
        from pyhorn_core.solver.wavefront import load_horn_geometry
        geo = load_horn_geometry(geometry)
        coords = geo["coords"]
        typer.echo(f"  Name: {geo.get('name', 'unknown')}")
        typer.echo(f"  Vertices: {len(coords)}")
        typer.echo(f"  x range: {coords[:,0].min():.4f} – {coords[:,0].max():.4f} m")
        typer.echo(f"  y range: {coords[:,1].min():.4f} – {coords[:,1].max():.4f} m")
        if geo.get("source_x") is not None:
            typer.echo(f"  Driver: ({geo['source_x']:.4f}, {geo['source_y']:.4f}) m")
        typer.echo("\nDry run complete. Geometry is valid.")
        raise typer.Exit(code=0)

    if simulate:
        from pyhorn_core.solver.wavefront import edit_horn_geometry_and_simulate
        result = edit_horn_geometry_and_simulate(
            yaml_path=geometry,
            output_yaml_path=output,
            output_png_path=None,
            frequency=frequency,
            grid_size=grid_size,
            title=title,
            figsize=figsize,
        )
        n_verts = len(result.get("coords", []))
        png_path = result.get("png_path", "the PNG")
        typer.secho(
            f"Editor closed. Wave simulation saved ({n_verts} vertices, {frequency} Hz) → {png_path}",
            fg=typer.colors.GREEN,
        )
        if result.get("saved"):
            typer.secho(f"  + edited geometry saved to {output}", fg=typer.colors.GREEN)
        return

    result = edit_horn_geometry_from_yaml(yaml_path=geometry, output_yaml_path=output, title=title, figsize=figsize)
    n_verts = len(result.get("coords", []))
    if result.get("saved"):
        typer.secho(f"Saved edited geometry ({n_verts} vertices) to {output}", fg=typer.colors.GREEN)
    else:
        if output is not None:
            typer.secho("Editor cancelled — geometry NOT saved (Escape was pressed).", fg=typer.colors.YELLOW)
        else:
            typer.secho(f"Editor closed — {n_verts} vertices in memory. Use --output to save.", fg=typer.colors.CYAN)
