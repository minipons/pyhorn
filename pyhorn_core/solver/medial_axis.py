"""
Backward-compatibility shim for pyhorn_core.solver.medial_axis.

All code has been moved to pyhorn_segment.segment.
This module re-exports everything from there so that existing imports
(e.g. ``from pyhorn_core.solver.medial_axis import generate_auto_segments``)
continue to work without modification.

When ``pyhorn_segment`` is not installed, all symbols are None — any caller
that tries to use them will get a clear AttributeError.
"""

try:
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
    _HAS_PYHORN_SEGMENT = True
except ImportError:
    _HAS_PYHORN_SEGMENT = False
    _names = [
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
    for _name in _names:
        globals()[_name] = None

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
