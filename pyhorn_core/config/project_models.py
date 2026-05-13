from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class HornProject:
    """Project metadata that wraps geometry and enclosure-specific overrides."""

    name: Optional[str] = None
    geometry_path: Optional[str] = None
    driver_coord: Optional[Tuple[float, float]] = None
    width: Optional[float] = None
    enclosure: Optional[Tuple[float, float]] = None
    thickness: float = 0.0
    material: Optional[str] = None
    notes: Optional[str] = None
    fold_plot_segments: Optional[List[Tuple[float, ...]]] = None
    rear_chamber: Optional["RearChamber"] = None
    throat_chamber: Optional["ThroatChamber"] = None
    vented_box: Optional["VentedBox"] = None
    passive_radiator: Optional["PassiveRadiator"] = None
    slavbas: Optional["SlavicBox"] = None
    sensitivity_db: Optional[np.ndarray] = None
