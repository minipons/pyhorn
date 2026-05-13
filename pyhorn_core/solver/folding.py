"""
Folded-horn geometry layout tools.

This module is a **backward-compat shim**: the actual implementation lives in
``pyhorn_fold/`` (extracted May 2, 2026). All public symbols are re-exported
from there unchanged, so existing callers (``pyhorn_cli``, other solvers) need
no modification.
"""
from pyhorn_fold import (
    _point_along_segment,
    _throat_attachment_point,
    _fold_segment_heights,
    _fold_path_polygons,
    _build_vertical_fold_path_from_lanes,
    _build_vertical_boundary_terminal_l_path,
    _build_vertical_fold_path_optimized,
    _build_fold_path,
    _reflect_lanes_from_mirrors,
    _segment_height_limit,
    throat_chamber_side_length,
    extrapolate_folded_horn,
)

__all__ = [
    "throat_chamber_side_length",
    "extrapolate_folded_horn",
    "_point_along_segment",
    "_throat_attachment_point",
    "_segment_height_limit",
    "_fold_segment_heights",
    "_fold_path_polygons",
    "_build_vertical_fold_path_from_lanes",
    "_build_vertical_boundary_terminal_l_path",
    "_build_vertical_fold_path_optimized",
    "_build_fold_path",
    "_reflect_lanes_from_mirrors",
]
