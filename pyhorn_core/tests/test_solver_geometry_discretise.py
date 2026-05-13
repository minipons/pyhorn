"""Unit tests for pyhorn.solver.geometry_discretise."""

import numpy as np
import pytest
from pyhorn_core.solver.geometry_discretise import (
    compute_bend_angles,
    compute_perpendicular_sections,
    discretise_geometry_aware,
)


class TestComputeBendAngles:
    """Tests for compute_bend_angles()."""

    def test_straight_line_zero_angle(self):
        """A straight line should give zero bend angle at interior points."""
        coords = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]
        angles = compute_bend_angles(coords)
        assert len(angles) == 1
        assert angles[0] == pytest.approx(0.0, abs=1e-9)

    def test_90_degree_turn(self):
        """A 90° turn should give π/2 radians."""
        coords = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1]]
        angles = compute_bend_angles(coords)
        assert angles[0] == pytest.approx(np.pi / 2, rel=1e-6)

    def test_180_degree_reversal(self):
        """A 180° reversal should give π radians."""
        coords = [[0.0, 0.0], [0.1, 0.0], [0.0, 0.0]]
        angles = compute_bend_angles(coords)
        assert angles[0] == pytest.approx(np.pi, rel=1e-6)

    def test_three_points_two_angles(self):
        """3 points → 1 interior → 1 bend angle."""
        coords = [[0.0, 0.0], [0.1, 0.05], [0.2, 0.1]]
        angles = compute_bend_angles(coords)
        assert len(angles) == 1

    def test_five_points_three_angles(self):
        """5 points → 3 interior points → 3 bend angles."""
        coords = [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.15, 0.05],
            [0.2, 0.05],
            [0.3, 0.0],
        ]
        angles = compute_bend_angles(coords)
        assert len(angles) == 3
        # All should be non-negative
        assert all(a >= 0 for a in angles)
        # None should exceed π
        assert all(a <= np.pi + 1e-9 for a in angles)

    def test_zero_length_segment_skipped(self):
        """Zero-length segments should produce 0.0 angle."""
        coords = [[0.0, 0.0], [0.0, 0.0], [0.1, 0.0]]
        angles = compute_bend_angles(coords)
        assert angles[0] == pytest.approx(0.0, abs=1e-9)

    def test_bend_angle_limited_to_pi(self):
        """Computed angle should be clipped to [0, π]."""
        coords = [[0.0, 0.0], [0.05, 0.0], [0.1, 0.0001]]  # tiny turn
        angles = compute_bend_angles(coords)
        assert 0 <= angles[0] <= np.pi

    def test_returns_list_of_floats(self):
        """Should return a list of plain float values."""
        coords = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1]]
        angles = compute_bend_angles(coords)
        assert isinstance(angles, list)
        assert all(isinstance(a, float) for a in angles)


class TestComputePerpendicularSections:
    """Tests for compute_perpendicular_sections()."""

    @pytest.fixture
    def simple_rectangle(self):
        """A 0.2 × 0.1 rectangle polygon."""
        from shapely.geometry import Polygon
        # 0.2 wide x 0.1 tall
        return Polygon([[0, 0], [0.2, 0], [0.2, 0.1], [0, 0.1]])

    @pytest.fixture
    def simple_triangle(self):
        """A right triangle polygon."""
        from shapely.geometry import Polygon
        return Polygon([[0, 0], [0.2, 0], [0.1, 0.1]])

    def test_returns_list_of_floats(self, simple_rectangle):
        """Should return a list of widths (floats)."""
        centerline = [[0.1, 0.0], [0.1, 0.1]]
        widths = compute_perpendicular_sections(simple_rectangle, centerline)
        assert isinstance(widths, list)
        assert all(isinstance(w, float) for w in widths)

    def test_length_matches_centerline_points(self, simple_rectangle):
        """Widths list length should equal number of centerline points."""
        centerline = [[0.1, 0.0], [0.1, 0.05], [0.1, 0.1]]
        widths = compute_perpendicular_sections(simple_rectangle, centerline)
        assert len(widths) == len(centerline)

    def test_width_positive_inside_polygon(self, simple_rectangle):
        """All widths should be positive for points inside the polygon."""
        centerline = [[0.1, 0.01], [0.1, 0.05], [0.1, 0.09]]
        widths = compute_perpendicular_sections(simple_rectangle, centerline)
        assert all(w > 0 for w in widths)

    def test_width_bounded_by_polygon(self, simple_rectangle):
        """No width should exceed the polygon's maximum extent."""
        centerline = [[0.1, 0.01], [0.1, 0.05], [0.1, 0.09]]
        widths = compute_perpendicular_sections(simple_rectangle, centerline)
        max_extent = max(simple_rectangle.bounds[2] - simple_rectangle.bounds[0],
                        simple_rectangle.bounds[3] - simple_rectangle.bounds[1])
        assert all(w <= max_extent for w in widths)

    def test_multiple_points_all_get_widths(self, simple_rectangle):
        """Multiple centerline points should each get a width."""
        centerline = [[0.1, 0.01], [0.1, 0.05], [0.1, 0.09]]
        widths = compute_perpendicular_sections(simple_rectangle, centerline)
        assert len(widths) == 3
        assert all(w > 0 for w in widths)


class TestDiscretiseGeometryAware:
    """Tests for discretise_geometry_aware()."""

    def test_returns_three_tuples(self):
        """Should return (segments, bends, bend_positions)."""
        segs, bends, b_pos = discretise_geometry_aware(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            bend_angles=[],
            n_per_segment=5,
        )
        assert isinstance(segs, list)
        assert isinstance(bends, list)
        assert isinstance(b_pos, list)

    def test_n_per_segment_controls_count(self):
        """n_per_segment=10 on one segment gives 10 sub-segments."""
        segs, _, _ = discretise_geometry_aware(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            n_per_segment=10,
        )
        assert len(segs) == 10

    def test_three_segments_gives_30_subsegments(self):
        """Three segments each with n_per_segment=10 gives 30 total."""
        segs, _, _ = discretise_geometry_aware(
            [(0.06, 0.07, 0.03), (0.07, 0.08, 0.04), (0.08, 0.09, 0.05)],
            width=0.2,
            n_per_segment=10,
        )
        assert len(segs) == 30

    def test_segments_are_length_area_fr_tuples(self):
        """Each segment should be (length, area, fr) 3-tuple."""
        segs, _, _ = discretise_geometry_aware(
            [(0.06, 0.07, 0.03, 5000.0)],
            width=0.2,
            n_per_segment=5,
        )
        for seg in segs:
            assert len(seg) == 3
            assert seg[0] > 0
            assert seg[1] > 0

    def test_bend_appended_at_area_mismatch(self):
        """A bend should be recorded when area doesn't match at junction."""
        segs, bends, b_pos = discretise_geometry_aware(
            [(0.06, 0.07, 0.03), (0.08, 0.09, 0.04)],  # end 0.07, start 0.08
            width=0.2,
            n_per_segment=5,
            bend_angles=[0.0],
        )
        assert len(bends) == 1
        assert len(b_pos) == 1

    def test_no_bend_when_areas_match(self):
        """No bend when the end area of one segment equals start of next."""
        segs, bends, b_pos = discretise_geometry_aware(
            [(0.06, 0.07, 0.03), (0.07, 0.09, 0.04)],  # end 0.07, start 0.07
            width=0.2,
            n_per_segment=5,
            bend_angles=[0.0],
        )
        assert len(bends) == 0
        assert len(b_pos) == 0

    def test_bend_includes_angle(self):
        """When recorded, the bend should include the angle_rad."""
        segs, bends, b_pos = discretise_geometry_aware(
            [(0.06, 0.07, 0.03), (0.08, 0.09, 0.04)],
            width=0.2,
            n_per_segment=5,
            bend_angles=[np.pi / 4],  # 45° bend
        )
        assert len(bends) == 1
        assert bends[0][2] == pytest.approx(np.pi / 4)

    def test_bend_positions_references_correct_subsegment(self):
        """bend_positions should index into the sub-segment list correctly."""
        segs, bends, b_pos = discretise_geometry_aware(
            [(0.06, 0.07, 0.03), (0.08, 0.09, 0.04)],
            width=0.2,
            n_per_segment=5,
            bend_angles=[0.0],
        )
        # With n_per_segment=5, first segment has indices 0-4, bend should be at 4
        assert b_pos[0] == 4
        assert b_pos[0] < len(segs)

    def test_raises_on_bend_angles_wrong_length(self):
        """bend_angles length must match len(segments) - 1."""
        with pytest.raises(ValueError, match="bend_angles length"):
            discretise_geometry_aware(
                [(0.06, 0.07, 0.03), (0.07, 0.08, 0.04)],
                width=0.2,
                bend_angles=[0.0, 0.0, 0.0],  # 3 values but only 2 segments
            )

    def test_without_width_areas_are_dim_values(self):
        """Without width, areas are the dim values directly."""
        segs, _, _ = discretise_geometry_aware(
            [(0.01, 0.02, 0.05)],
            width=None,
            n_per_segment=5,
        )
        # area_start = 0.01, area_end = 0.02
        # First subsegment avg: (0.01 + 0.012)/2 = 0.011
        assert segs[0][1] == pytest.approx(0.011, rel=1e-6)

    def test_fr_preserved_from_4th_element(self):
        """4th element in segment tuple should become fr in sub-segments."""
        segs, _, _ = discretise_geometry_aware(
            [(0.06, 0.07, 0.03, 5000.0)],
            width=0.2,
            n_per_segment=5,
        )
        for seg in segs:
            assert seg[2] == pytest.approx(5000.0)

    def test_default_fr_is_zero(self):
        """Without 4th element, fr defaults to 0.0."""
        segs, _, _ = discretise_geometry_aware(
            [(0.06, 0.07, 0.03)],
            width=0.2,
            n_per_segment=5,
        )
        for seg in segs:
            assert seg[2] == pytest.approx(0.0)

    def test_bends_tuple_is_area_area_angle(self):
        """Each bend should be (area_before, area_after, angle_rad)."""
        segs, bends, _ = discretise_geometry_aware(
            [(0.06, 0.07, 0.03), (0.08, 0.09, 0.04)],
            width=0.2,
            n_per_segment=5,
            bend_angles=[0.5],
        )
        for bend in bends:
            assert len(bend) == 3
            assert bend[0] > 0  # area_before
            assert bend[1] > 0  # area_after
            assert bend[2] >= 0  # angle_rad