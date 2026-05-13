"""pyhorn_cli CLI entry point — registers all command groups."""

import typer

from .core_commands import calculate, compare, derive_ts
from .design_wizard import resize_wizard, hornresp, chamber_wizard, segment_wizard, synthesis_wizard
from .horn_commands import tapped_horn, throat_adapter, auto_segment, diagnose_spl, driver_front_volume
from .interactive_commands import wavefront_edit
from .optimize_commands import optimize, fold_optimized

app = typer.Typer(help="pyhornresp: Headless Acoustic Simulation CLI")


def _register_commands():
    app.command(name="calculate")(calculate)
    app.command(name="compare")(compare)
    app.command(name="derive-ts")(derive_ts)
    app.command(name="resize-wizard")(resize_wizard)
    app.command(name="hornresp")(hornresp)
    app.command(name="chamber-wizard")(chamber_wizard)
    app.command(name="segment-wizard")(segment_wizard)
    app.command(name="synthesis-wizard")(synthesis_wizard)
    app.command(name="tapped-horn")(tapped_horn)
    app.command(name="throat-adapter")(throat_adapter)
    app.command(name="auto-segment")(auto_segment)
    app.command(name="diagnose-spl")(diagnose_spl)
    app.command(name="driver-front-volume")(driver_front_volume)
    app.command(name="optimize")(optimize)
    app.command(name="fold-optimized")(fold_optimized)
    app.command(name="wavefront-edit")(wavefront_edit)


_register_commands()


if __name__ == "__main__":
    app()
