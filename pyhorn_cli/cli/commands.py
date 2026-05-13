"""Backwards-compatibility shim — re-exports from the new modular structure.

``main.py`` imports from here, so all names must remain available.
"""

# Import app (with all decorators applied) and individual commands from cli.py
from pyhorn_cli.cli.cli import (
    app as commands_app,
    calculate,
    compare,
    derive_ts,
    resize_wizard,
    hornresp,
    chamber_wizard,
    segment_wizard,
    synthesis_wizard,
    tapped_horn,
    throat_adapter,
    auto_segment,
    diagnose_spl,
    optimize,
    driver_front_volume,
)

__all__ = [
    "commands_app",
    "calculate",
    "compare",
    "derive_ts",
    "resize_wizard",
    "hornresp",
    "chamber_wizard",
    "segment_wizard",
    "synthesis_wizard",
    "tapped_horn",
    "throat_adapter",
    "auto_segment",
    "diagnose_spl",
    "optimize",
    "driver_front_volume",
]
