"""Compatibility facade for pyhorn configuration dataclasses.

Canonical definitions now live in the focused modules under `pyhorn_core.config`:

- `driver_models.py`
- `horn_models.py`
- `chamber_models.py`
- `project_models.py`

Importing from `pyhorn_core.config.models` remains supported for backward compatibility.
"""

from pyhorn_core.config.chamber_models import (
    PassiveRadiator,
    RearChamber,
    SlavicBox,
    ThroatAdapter,
    ThroatChamber,
    VentedBox,
)
from pyhorn_core.config.driver_models import DriverSpecs
from pyhorn_core.config.horn_models import (
    CompoundChamber,
    HornGeometry,
    Section,
    TappedHornGeometry,
)
from pyhorn_core.config.project_models import HornProject

__all__ = [
    "CompoundChamber",
    "DriverSpecs",
    "HornGeometry",
    "HornProject",
    "PassiveRadiator",
    "RearChamber",
    "Section",
    "SlavicBox",
    "TappedHornGeometry",
    "ThroatAdapter",
    "ThroatChamber",
    "VentedBox",
]
