"""
Backward-compatibility shim for pyhorn_core.solver.medial_axis.

All code has been moved to pyhorn_segment.segment.
This module re-exports everything from there so that existing imports
(e.g. ``from pyhorn_core.solver.medial_axis import generate_auto_segments``)
continue to work without modification.
"""

# Re-export the public API from the new pyhorn_segment package
from pyhorn_segment import (
    generate_auto_segments,
    rectangular_segments_to_sections,
    _reduce_stair_points,
    _remove_duplicate_stations,
    _distance_to_nearest_wall,
    _perpendicular_width_at_point,
    _build_wall_lines,
    _infer_profile_type,
    _generate_graph_method,
    _generate_voronoi_method,
)

__all__ = [
    "generate_auto_segments",
    "rectangular_segments_to_sections",
    "_reduce_stair_points",
    "_remove_duplicate_stations",
    "_distance_to_nearest_wall",
    "_perpendicular_width_at_point",
    "_build_wall_lines",
    "_infer_profile_type",
    "_generate_graph_method",
    "_generate_voronoi_method",
]
