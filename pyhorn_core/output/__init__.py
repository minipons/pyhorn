"""Output generation module for pyhornresp.

Public API
----------
Export functions:
    export_to_csv   → export SPL data to CSV
    export_to_json  → export SPL data to JSON
    export_to_frd   → export SPL data to .frd (REW/ARTA format)

Plot functions:
    plot_simulation_results  → main acoustic response plot (SPL, impedance, etc.)
    plot_polar_response      → polar directivity plot
    plot_horn_3d            → 3D horn geometry plot
    plot_horn_2d_folded     → 2D folded horn path diagram
    plot_waterfall          → waterfall spectrogram
    plot_throat_adapter_profile  → throat adapter geometry profile
    plot_impulse_step       → impulse and step response
"""

from pyhorn_core.output.exporter import export_to_csv, export_to_json, export_to_frd
from pyhorn_core.output.plotter import (
    plot_simulation_results,
    plot_polar_response,
    plot_horn_3d,
    plot_horn_2d_folded,
    plot_waterfall,
    plot_throat_adapter_profile,
    plot_impulse_step,
)
