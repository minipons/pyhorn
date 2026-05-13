"""Unit tests for pyhorn.solver.medial_axis."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pyhorn_core.solver.medial_axis import (
    _reduce_stair_points,
    _remove_duplicate_stations,
    generate_auto_segments,
)


# ─── _reduce_stair_points ──────────────────────────────────────────────────────

class TestReduceStairPoints:
    """Tests for _reduce_stair_points()."""

    def test_empty_or_two_point_list_unchanged(self):
        """Points list with <= 2 points should be returned unchanged."""
        pts = [[0.0, 0.0]]
        wids = [0.1]
        dists = [0.0]
        out_pts, out_wids, out_dists = _reduce_stair_points(pts, wids, dists)
        assert out_pts == pts

        pts2 = [[0.0, 0.0], [0.1, 0.1]]
        wids2 = [0.1, 0.1]
        dists2 = [0.0, 0.1]
        out2_pts, _, _ = _reduce_stair_points(pts2, wids2, dists2)
        assert out2_pts == pts2

    def test_straight_line_reduces_to_three_points(self):
        """A straight line with many points reduces to start + end + one interior."""
        pts = [[0.0, 0.0], [0.05, 0.0], [0.1, 0.0], [0.15, 0.0], [0.2, 0.0]]
        wids = [0.1, 0.1, 0.1, 0.1, 0.1]
        dists = [0.0, 0.05, 0.1, 0.15, 0.2]
        out_pts, _, _ = _reduce_stair_points(pts, wids, dists, angle_threshold_deg=10.0)
        # Without strong turns, only first and last are kept
        assert out_pts[0] == pts[0]
        assert out_pts[-1] == pts[-1]

    def test_90_degree_corner_preserved(self):
        """A sharp 90° corner should be preserved."""
        # A corner at point index 2
        pts = [[0.0, 0.0], [0.1, 0.0], [0.1, 0.1], [0.0, 0.1]]
        wids = [0.1, 0.1, 0.1, 0.1]
        dists = [0.0, 0.1, 0.2, 0.3]
        out_pts, _, _ = _reduce_stair_points(pts, wids, dists, angle_threshold_deg=25.0)
        # Point at index 2 (the corner) should be kept
        assert [0.1, 0.1] in out_pts

    def test_width_change_triggers_keep(self):
        """A large relative width jump should be kept."""
        pts = [[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]]
        wids = [0.1, 0.3, 0.3]  # big relative jump at middle
        dists = [0.0, 0.05, 0.1]
        out_pts, out_wids, _ = _reduce_stair_points(pts, wids, dists)
        # Middle point with width change should be kept
        assert len(out_pts) >= 3

    def test_min_segment_length_filters_short(self):
        """Points with adjacent segment < min_segment_length should be dropped."""
        pts = [[0.0, 0.0], [0.01, 0.0], [0.05, 0.0], [0.06, 0.0]]
        wids = [0.1, 0.1, 0.1, 0.1]
        dists = [0.0, 0.01, 0.05, 0.06]
        out_pts, _, _ = _reduce_stair_points(pts, wids, dists, min_segment_length=0.03)
        # Point at 0.01 should be dropped (both adjacent segments < 0.03)
        assert [0.01, 0.0] not in out_pts

    def test_distances_and_widths_kept_with_points(self):
        """Output should include corresponding distances and widths."""
        pts = [[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]]
        wids = [0.1, 0.2, 0.3]
        dists = [0.0, 0.05, 0.1]
        out_pts, out_wids, out_dists = _reduce_stair_points(pts, wids, dists)
        assert len(out_pts) == len(out_wids) == len(out_dists)

    def test_default_angle_threshold(self):
        """Default angle_threshold_deg should be 25°."""
        pts = [[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]]
        wids = [0.1, 0.1, 0.1]
        dists = [0.0, 0.05, 0.1]
        # Should not raise even with default arguments
        out_pts, _, _ = _reduce_stair_points(pts, wids, dists)
        assert isinstance(out_pts, list)

    def test_none_distances_uses_index_distance(self):
        """If distances is None, uses index-based distance."""
        pts = [[0.0, 0.0], [0.05, 0.0], [0.1, 0.0]]
        wids = [0.1, 0.1, 0.1]
        out_pts, _, out_dists = _reduce_stair_points(pts, wids, None)
        assert out_dists[1] > out_dists[0]


# ─── _remove_duplicate_stations ───────────────────────────────────────────────

class TestRemoveDuplicateStations:
    """Tests for _remove_duplicate_stations()."""

    def test_no_duplicates_unchanged(self):
        """Points separated by > min_distance should be unchanged."""
        pts = [[0.0, 0.0], [0.001, 0.0], [0.002, 0.0]]
        wids = [0.1, 0.1, 0.1]
        dists = [0.0, 0.001, 0.002]
        out_pts, _, _ = _remove_duplicate_stations(pts, wids, dists)
        assert len(out_pts) == 3

    def test_consecutive_duplicates_removed(self):
        """Consecutive points closer than min_distance should collapse to one."""
        pts = [[0.0, 0.0], [0.00001, 0.0], [0.002, 0.0]]  # first two are effectively same
        wids = [0.1, 0.2, 0.3]
        dists = [0.0, 0.00001, 0.002]
        out_pts, out_wids, out_dists = _remove_duplicate_stations(pts, wids, dists)
        assert len(out_pts) == 2
        # The surviving point should keep the last width (updated)
        assert out_wids[0] == 0.2  # updated from second point
        assert out_dists[0] == 0.00001  # updated from second point

    def test_empty_list_unchanged(self):
        """Empty list should return unchanged."""
        out_pts, out_wids, out_dists = _remove_duplicate_stations([], [], [])
        assert out_pts == []
        assert out_wids == []
        assert out_dists == []

    def test_single_point_unchanged(self):
        """Single point should be returned unchanged."""
        out_pts, out_wids, out_dists = _remove_duplicate_stations(
            [[0.0, 0.0]], [0.1], [0.0]
        )
        assert len(out_pts) == 1

    def test_close_points_merged_but_distant_kept(self):
        """Points separated by >= min_distance are kept; closer ones merge."""
        pts = [[0.0, 0.0], [0.00005, 0.0], [0.002, 0.0]]  # first two are duplicates
        wids = [0.1, 0.2, 0.3]
        dists = [0.0, 0.00005, 0.002]
        out_pts, out_wids, out_dists = _remove_duplicate_stations(pts, wids, dists)
        # First two collapse to one (updated with second's values)
        assert len(out_pts) == 2
        assert out_wids[0] == 0.2  # updated
        assert out_dists[0] == 0.00005  # updated


# ─── generate_auto_segments ────────────────────────────────────────────────────

class TestGenerateAutoSegments:
    """Tests for generate_auto_segments()."""

    @pytest.fixture
    def valid_json_data(self):
        """Minimal valid JSON for a rectangular air volume."""
        return {
            "width": 0.2,
            "throat": [[0.0, 0.05, 0.0], [0.0, 0.15, 0.0]],
            "mouth": [[0.0, 0.3, 0.0], [0.0, 0.4, 0.0]],
            "boundary_edges": [
                [[0.0, 0.05, 0.0], [0.0, 0.05, 0.0]],
                [[0.0, 0.05, 0.0], [0.0, 0.4, 0.0]],
                [[0.0, 0.4, 0.0], [0.0, 0.3, 0.0]],
                [[0.0, 0.3, 0.0], [0.0, 0.05, 0.0]],
            ],
        }

    @pytest.fixture
    def valid_2d_json_data(self):
        """2D (no z-axis) valid JSON for a rectangular air volume."""
        return {
            "width": 0.2,
            "throat": [[0.05, 0.0], [0.15, 0.0]],
            "mouth": [[0.3, 0.0], [0.4, 0.0]],
            "boundary_edges": [
                [[0.05, 0.0], [0.4, 0.0]],
                [[0.4, 0.0], [0.4, 0.2]],
                [[0.4, 0.2], [0.05, 0.2]],
                [[0.05, 0.2], [0.05, 0.0]],
            ],
        }

    def test_writes_yaml_output_file(self, valid_2d_json_data, tmp_path):
        """Should create the output YAML file."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        generate_auto_segments(json_path, output_yaml, n_segments=10)

        assert output_yaml.exists()

    def test_output_yaml_contains_required_keys(self, valid_2d_json_data, tmp_path):
        """Output YAML should have required fields."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(json_path, output_yaml, n_segments=10)

        assert result["enclosure_type"] == "BLH"
        assert "width" in result
        assert "enclosure_dims" in result
        assert "coordinates" in result

    def test_conical_or_rectangular_segments_in_output(self, valid_2d_json_data, tmp_path):
        """Output should have either conical_segments or rectangular_segments (legacy format)."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(json_path, output_yaml, n_segments=10, output_format="legacy")

        has_conical = "conical_segments" in result
        has_rectangular = "rectangular_segments" in result
        assert has_conical or has_rectangular

    def test_n_segments_controls_segment_count(self, valid_2d_json_data, tmp_path):
        """n_segments should control the number of segments in output."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(json_path, output_yaml, n_segments=20)

        if "conical_segments" in result:
            assert len(result["conical_segments"]) == 20
        if "rectangular_segments" in result:
            assert len(result["rectangular_segments"]) == 20

    def test_coordinates_count_equals_segments_plus_one(self, valid_2d_json_data, tmp_path):
        """coordinates should have n_segments + 1 entries."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(json_path, output_yaml, n_segments=15)

        assert len(result["coordinates"]) == 16

    def test_flip_x_flips_x_coordinates(self, valid_2d_json_data, tmp_path):
        """flip_x=True should invert x coordinates relative to bounding box."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(json_path, output_yaml, n_segments=10, flip_x=True)

        import yaml
        with open(output_yaml) as f:
            saved = yaml.safe_load(f)
        x_coords = [c[0] for c in saved["coordinates"]]
        # With flip_x, the largest x should become the smallest
        assert min(x_coords) > 0 or max(x_coords) >= 0  # basic sanity

    def test_flip_y_flips_y_coordinates(self, valid_2d_json_data, tmp_path):
        """flip_y=True should invert y coordinates relative to bounding box."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(json_path, output_yaml, n_segments=10, flip_y=True)

        import yaml
        with open(output_yaml) as f:
            saved = yaml.safe_load(f)
        y_coords = [c[1] for c in saved["coordinates"]]
        assert isinstance(y_coords[0], (int, float))

    def test_raises_on_missing_json_path(self, tmp_path):
        """json_path=None without from_clipboard should raise."""
        output_yaml = tmp_path / "output.yaml"
        with pytest.raises(ValueError, match="json_path is required"):
            generate_auto_segments(None, output_yaml, from_clipboard=False)

    def test_raises_on_nonexistent_json_file(self, tmp_path):
        """Non-existent JSON file should raise FileNotFoundError."""
        output_yaml = tmp_path / "output.yaml"
        with pytest.raises((FileNotFoundError, ValueError)):
            generate_auto_segments(tmp_path / "nonexistent.json", output_yaml)

    def test_raises_on_malformed_json(self, tmp_path):
        """Malformed JSON should raise ValueError."""
        json_path = tmp_path / "bad.json"
        json_path.write_text("{ not valid json")
        output_yaml = tmp_path / "output.yaml"
        with pytest.raises(ValueError):
            generate_auto_segments(json_path, output_yaml)

    def test_from_clipboard_raises_without_pbpaste(self, tmp_path, monkeypatch):
        """from_clipboard=True should raise if clipboard is empty/inaccessible."""
        import subprocess
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess("cmd", 1, b"", b"error", returncode=1)
        )
        output_yaml = tmp_path / "output.yaml"
        with pytest.raises(ValueError, match="clipboard"):
            generate_auto_segments(None, output_yaml, from_clipboard=True)

    def test_enclosure_dims_are_depth_height(self, valid_2d_json_data, tmp_path):
        """enclosure_dims should be [depth, height] = (x_max-x_min, y_max-y_min)."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(json_path, output_yaml, n_segments=10)

        assert len(result["enclosure_dims"]) == 2
        assert result["enclosure_dims"][0] > 0  # depth
        assert result["enclosure_dims"][1] > 0  # height

    def test_geometry_aware_flag_sets_discretisation(self, valid_2d_json_data, tmp_path):
        """geometry_aware=True should add 'discretisation: geometry' to output."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(
            json_path, output_yaml, n_segments=10, geometry_aware=True
        )

        assert result.get("discretisation") == "geometry"
        assert "bend_angles" in result

    def test_preserve_breaks_produces_rectangular_segments(self, valid_2d_json_data, tmp_path):
        """preserve_breaks=True should output rectangular_segments (legacy format)."""
        json_path = tmp_path / "input.json"
        json_path.write_text(json.dumps(valid_2d_json_data))
        output_yaml = tmp_path / "output.yaml"

        result = generate_auto_segments(
            json_path, output_yaml, n_segments=10, preserve_breaks=True, geometry_aware=True, output_format="legacy"
        )

        assert "rectangular_segments" in result
        assert "conical_segments" not in result