"""
Backward-compatibility shim for pyhorn_core.solver.wavefront.

All code has been moved to pyhorn_wavefront.wavefront.
This module re-exports everything from there so that existing imports
(e.g. ``from pyhorn_core.solver.wavefront import WavefrontGrid``)
continue to work without modification.
"""

from pyhorn_wavefront import (
    WavefrontGrid,
    solve_2d_wave,
    solve_2d_wave_pml,
    boundary_condition_mask,
    load_horn_geometry,
    edit_horn_geometry,
    edit_horn_geometry_from_yaml,
    edit_horn_geometry_and_simulate,
    pml_damping_mask,
    compute_pressure_field,
    ka_warning,
    plot_wavefront,
    plot_pressure_amplitude,
    plot_amplitude_db,
    animate_wave_propagation,
    plot_animation_frames,
    plot_wavefront_polar,
    solve_2d_wave_time_domain,
    solve_2d_wave_time_domain_pml,
    WavefrontGrid_animate,
)

__all__ = [
    "WavefrontGrid",
    "solve_2d_wave",
    "solve_2d_wave_pml",
    "boundary_condition_mask",
    "load_horn_geometry",
    "edit_horn_geometry",
    "edit_horn_geometry_from_yaml",
    "edit_horn_geometry_and_simulate",
    "pml_damping_mask",
    "compute_pressure_field",
    "ka_warning",
    "plot_wavefront",
    "plot_pressure_amplitude",
    "plot_amplitude_db",
    "animate_wave_propagation",
    "plot_animation_frames",
    "plot_wavefront_polar",
    "solve_2d_wave_time_domain",
    "solve_2d_wave_time_domain_pml",
    "WavefrontGrid_animate",
]
