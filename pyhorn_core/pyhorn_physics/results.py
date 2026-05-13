from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SimulationResult:
    freqs: np.ndarray
    spl: np.ndarray
    impedance: np.ndarray
    excursion: np.ndarray
    segments: Optional[List[Tuple[float, ...]]] = None
    ib_spl: Optional[np.ndarray] = None
    direct_spl: Optional[np.ndarray] = None
    horn_spl: Optional[np.ndarray] = None
    group_delay: Optional[np.ndarray] = None
    group_delay_per_period: Optional[np.ndarray] = None
    phase: Optional[np.ndarray] = None
    pressure: Optional[np.ndarray] = None
    throat_impedance: Optional[np.ndarray] = None
    impedance_phase_deg: Optional[np.ndarray] = None
    segment_widths: Optional[List[float]] = None
    numerical_artifacts: Optional[List[float]] = None
    efficiency_pct: Optional[np.ndarray] = None
    electrical_input_power: Optional[np.ndarray] = None
    off_axis_spl: Optional[np.ndarray] = None
    off_axis_angles: Optional[np.ndarray] = None
    radiation_angle: Optional[float] = None
    fdd_enabled: bool = False
    fdd_di: Optional[np.ndarray] = None
    direction_index: Optional[np.ndarray] = None
    finite_horn_charged: bool = False
    second_tone_distortion: Optional[np.ndarray] = None
    thermal_compression_db: Optional[np.ndarray] = None
    spl_notched: Optional[np.ndarray] = None
    room_gain_db: Optional[np.ndarray] = None
    room_type: Optional[str] = None
    cone_velocity: Optional[np.ndarray] = None
    cone_acceleration: Optional[np.ndarray] = None
    diaphragm_pressure_total: Optional[np.ndarray] = None
    diaphragm_pressure_horn_side: Optional[np.ndarray] = None
    diaphragm_pressure_direct_side: Optional[np.ndarray] = None
    particle_velocity_throat: Optional[np.ndarray] = None
    particle_velocity_mouth: Optional[np.ndarray] = None
    particle_velocity_port: Optional[np.ndarray] = None
    futtrup_gdlimit: Optional[np.ndarray] = None
    acoustic_power: Optional[np.ndarray] = None
    spl_power_based: Optional[np.ndarray] = None
