"""pyhorn_core — Acoustic Horn Simulator Core Library.

Headless acoustic horn loudspeaker simulator — core physics and numerics only,
no CLI or UI dependencies.

Main entry points
─────────────────
from pyhorn_core import parse_driver_specs, parse_horn_geometry, horn_response

driver = parse_driver_specs("path/to/driver.yaml")
horn   = parse_horn_geometry("path/to/horn.yaml")

# For project files that reference separate geometry/geometry files:
from pyhorn_core import parse_horn_project
"""

from pyhorn_core.config.parser import (
    parse_driver_specs,
    parse_horn_geometry,
    parse_horn_project,  # noqa: F401
)

# Re-export horn_response from the backward-compat solver.models shim so that
# `from pyhorn_core import horn_response` also works.
from pyhorn_core.solver.models import horn_response  # noqa: F401
