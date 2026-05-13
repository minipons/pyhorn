"""
Geometry-aware discretisation utilities for horn centre-line analysis.

Provides tools to translate a continuous horn centre-line (polygon boundary +
parametric path) into a set of discrete cross-sections with physically
meaningful widths and bend angles.

Main functions
~~~~~~~~~~~~~~
``compute_perpendicular_sections``
    Intersect perpendicular lines through each centre-line point with the
    polygon interior to get true cross-section widths (m).  Falls back to
    twice the distance-to-boundary when the intersection is degenerate.

``compute_bend_angles``
    Angle (radians) between consecutive centre-line tangent directions at
    each interior point.  Zero means straight, π means a 180° reversal.

``discretise_geometry_aware``
    Combines conical segment data with geometry-computed widths and bend
    angles to produce a uniform or adaptive discretisation of the horn path.

These functions are used by the synthesis wizard and the geometry export
pipeline to ensure that discretised horn geometry faithfully reflects the
as-built cabinet shape even when the path includes bends and non-uniform
expansion.
"""

import numpy as np
from shapely.geometry import (
    Polygon,
    LineString,
    Point,
    MultiLineString,
    GeometryCollection,
)
from typing import List, Tuple, Optional


def compute_perpendicular_sections(
    poly: Polygon, centerline_points: List[List[float]]
) -> List[float]:
    """
    Compute true cross-section widths by intersecting perpendicular lines
    with the polygon interior at each centerline point.

    Returns a list of widths (m), one per centerline point.
    """
    pts = np.array(centerline_points)
    boundary = poly.boundary
    n = len(pts)
    widths = []

    for i in range(n):
        # Tangent direction from neighbors
        if i == 0:
            tangent = pts[1] - pts[0]
        elif i == n - 1:
            tangent = pts[-1] - pts[-2]
        else:
            tangent = pts[i + 1] - pts[i - 1]

        t_len = np.linalg.norm(tangent)
        if t_len < 1e-10:
            widths.append(0.0)
            continue

        tangent = tangent / t_len
        normal = np.array([-tangent[1], tangent[0]])

        # Build a long perpendicular line through this point
        center = pts[i]
        extent = boundary.length  # generous extent
        p1 = center + normal * extent
        p2 = center - normal * extent
        perp_line = LineString([p1, p2])

        # Intersect with polygon interior and keep the local slice that
        # contains (or is nearest to) the centerline point.
        intersection = perp_line.intersection(poly)

        if intersection.is_empty:
            # Fallback: use distance to boundary (Voronoi-style)
            widths.append(boundary.distance(Point(center)) * 2)
            continue

        line_slices = []
        if intersection.geom_type == "LineString":
            line_slices = [intersection]
        elif isinstance(intersection, MultiLineString):
            line_slices = [
                g
                for g in intersection.geoms
                if isinstance(g, LineString) and g.length > 0.0
            ]
        elif isinstance(intersection, GeometryCollection):
            line_slices = [
                g
                for g in intersection.geoms
                if isinstance(g, LineString) and g.length > 0.0
            ]

        if line_slices:
            cpt = Point(center)
            local_slice = min(line_slices, key=lambda seg: seg.distance(cpt))
            widths.append(float(local_slice.length))
            continue

        # Fallback for degenerate intersections.
        widths.append(boundary.distance(Point(center)) * 2)

    return widths


def compute_bend_angles(coordinates: List[List[float]]) -> List[float]:
    """
    Compute bend angles at each interior coordinate point.
    Returns n-2 bend angles in radians (one per interior point).
    0 = straight, pi = 180° U-turn.
    """
    pts = np.array(coordinates)
    angles = []

    for i in range(1, len(pts) - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]

        len1 = np.linalg.norm(v1)
        len2 = np.linalg.norm(v2)

        if len1 < 1e-10 or len2 < 1e-10:
            angles.append(0.0)
            continue

        cos_angle = np.dot(v1, v2) / (len1 * len2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # Angle between consecutive centerline directions.
        # Straight continuation -> 0, 180 degree reversal -> pi.
        bend_angle = np.arccos(cos_angle)
        angles.append(float(bend_angle))

    return angles


def discretise_geometry_aware(
    conical_segments: List[Tuple[float, ...]],
    width: Optional[float],
    bend_angles: Optional[List[float]] = None,
    n_per_segment: int = 10,
) -> Tuple[List[Tuple[float, ...]], List[Tuple[float, float, float]], List[int]]:
    """
    Geometry-aware discretisation of conical segments with correct bend placement
    and angle information.

    Returns:
        segments: List of (length, area, fr) for each sub-segment
        bends: List of (area_before, area_after, angle_rad) at each junction
        bend_positions: List of sub-segment indices after which each bend is inserted
    """
    if bend_angles is not None and len(bend_angles) != len(conical_segments) - 1:
        raise ValueError(
            f"bend_angles length ({len(bend_angles)}) must equal "
            f"len(conical_segments) - 1 ({len(conical_segments) - 1})"
        )

    segments = []
    bends = []
    bend_positions = []

    last_area_end = None
    sub_segment_count = 0

    # bend_angles has n-2 entries (one per interior coordinate),
    # which maps to junctions between conical_segments[0]-[1], [1]-[2], etc.
    # So bend_angles[i] corresponds to the junction after conical_segments[i].

    for seg_idx, seg in enumerate(conical_segments):
        if len(seg) < 3:
            raise ValueError(
                "Each conical segment must contain at least start, end, and length"
            )
        if len(seg) >= 4:
            dim_start, dim_end, length, fr = seg[0], seg[1], seg[2], seg[3]
        else:
            dim_start, dim_end, length = seg[0], seg[1], seg[2]
            fr = 0.0

        if width is not None:
            area_start = dim_start * width
            area_end = dim_end * width
        else:
            area_start = dim_start
            area_end = dim_end

        # Preserve a geometric bend even when the cross-sectional area stays
        # continuous across the junction. Without this, constant-area folds are
        # silently treated as straight ducts.
        if last_area_end is not None:
            angle = 0.0
            if bend_angles and seg_idx - 1 < len(bend_angles):
                angle = bend_angles[seg_idx - 1]

            area_jump = abs(last_area_end - area_start) > 1e-6
            has_bend = angle > 1e-6
            if area_jump or has_bend:
                bends.append((last_area_end, area_start, angle))
                # Insert bend after the last sub-segment of the previous conical section
                bend_positions.append(sub_segment_count - 1)

        # Discretise this section
        dx = length / n_per_segment
        x = np.linspace(0, length, n_per_segment + 1)
        A_x = area_start + (area_end - area_start) * (x / length)

        for i in range(n_per_segment):
            A_avg = (A_x[i] + A_x[i + 1]) / 2.0
            segments.append((dx, float(A_avg), float(fr)))
            sub_segment_count += 1

        last_area_end = area_end

    return segments, bends, bend_positions
