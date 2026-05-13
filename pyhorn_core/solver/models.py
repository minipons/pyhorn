"""
pyhorn_core.solver.models — re-export shim.

All orchestrator functions and the SimulationResult dataclass have been moved to
pyhorn_physics.orchestrators. This module is kept as a backward-compat re-export
shim so that existing imports (e.g. from pyhorn_core.solver.models) continue to work.
"""
from __future__ import annotations

# Re-export everything from the canonical location
from pyhorn_core.pyhorn_physics.orchestrators import (
    SimulationResult,
    horn_response,
    horn_response_tapped,
    horn_response_compound,
    compute_thermal_power_compression,
    _horn_response_impl,
    _run_horn_response_internal,
    _pressure_to_spl,
    _detect_numerical_artifacts,
    _compute_second_tone_distortion,
    infinite_baffle_response,
)

# Re-export ALL physics primitives so that `from pyhorn_core.solver import models`
# still provides the full physics API (backward compat for tests and legacy code).
# These were migrated to pyhorn_physics/__init__.py in the models.py decomposition.
from pyhorn_core.pyhorn_physics import (
    RHO, C, Z0,
    _miki_factors,
    radiation_impedance,
    _circular_piston_radiation_impedance,
    tube_segment_tmatrix,
    area_step_tmatrix,
    bend_tmatrix,
    compliance_tmatrix,
    rear_chamber_impedance,
    vented_box_impedance,
    passive_radiator_impedance,
    slavbas_impedance,
    transmission_line_impedance,
    throat_adapter_tmatrix,
    cascade,
    _merge_small_throat_segments,
    _le_freq_dependent,
    _lossy_le_impedance,
    _driver_impedance,
    _velocity,
    _displacement,
    _excursion,
    _is_single_segment_horn,
    _compute_second_tone_distortion as _distortion_physics,
    _fdd_directivity_index,
    _fdd_off_axis_spl,
    _fdd_radiation_angle,
    _apply_notch_filter,
    _smooth_spl_near_artifacts,
)

__all__ = [
    # Orchestrators
    "SimulationResult",
    "horn_response",
    "horn_response_tapped",
    "horn_response_compound",
    "compute_thermal_power_compression",
    "_horn_response_impl",
    "_run_horn_response_internal",
    "_pressure_to_spl",
    "_detect_numerical_artifacts",
    "_compute_second_tone_distortion",
    "infinite_baffle_response",
    # Physics constants
    "RHO", "C", "Z0",
    # Physics primitives
    "_miki_factors",
    "radiation_impedance",
    "_circular_piston_radiation_impedance",
    "tube_segment_tmatrix",
    "area_step_tmatrix",
    "bend_tmatrix",
    "compliance_tmatrix",
    "rear_chamber_impedance",
    "vented_box_impedance",
    "passive_radiator_impedance",
    "slavbas_impedance",
    "transmission_line_impedance",
    "throat_adapter_tmatrix",
    "cascade",
    "_merge_small_throat_segments",
    "_le_freq_dependent",
    "_lossy_le_impedance",
    "_driver_impedance",
    "_velocity",
    "_displacement",
    "_excursion",
    "_is_single_segment_horn",
    "_distortion_physics",
    "_fdd_directivity_index",
    "_fdd_off_axis_spl",
    "_fdd_radiation_angle",
    "_apply_notch_filter",
    "_smooth_spl_near_artifacts",
]
