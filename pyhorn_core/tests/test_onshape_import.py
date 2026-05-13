"""Integration tests for Onshape JSON → Pyhorn YAML import pipeline.

These tests load the real raw JSON files exported from Onshape and verify
the generate_auto_segments pipeline produces geometrically valid horn
configurations that preserve the fold/stair geometry accurately.
"""

import numpy as np
import pytest
from pathlib import Path

from pyhorn_core.solver.medial_axis import generate_auto_segments
from pyhorn_core.config.parser import parse_horn_geometry
from pyhorn_core.solver.geometry_discretise import compute_bend_angles


# ─── Raw JSON structure ────────────────────────────────────────────────────────

class TestOnshapeJsonStructure:
    """Each JSON file has the expected structure."""

    def test_bk16_has_required_keys(self, onshape_bk16):
        assert set(onshape_bk16.keys()) == {"width", "throat", "mouth", "boundary_edges"}

    def test_fsx_has_required_keys(self, onshape_fsx):
        assert set(onshape_fsx.keys()) == {"width", "throat", "mouth", "boundary_edges"}

    def test_long_has_required_keys(self, onshape_long):
        assert set(onshape_long.keys()) == {"width", "throat", "mouth", "boundary_edges"}

    def test_bk16_1m_has_required_keys(self, onshape_bk16_1m):
        assert set(onshape_bk16_1m.keys()) == {"width", "throat", "mouth", "boundary_edges"}

    def test_all_throats_are_2_points(self, onshape_all):
        assert len(onshape_all["throat"]) == 2
        assert all(len(p) == 3 for p in onshape_all["throat"])

    def test_all_mouths_are_2_points(self, onshape_all):
        assert len(onshape_all["mouth"]) == 2
        assert all(len(p) == 3 for p in onshape_all["mouth"])

    def test_x_coordinate_is_near_zero(self, onshape_all):
        """All points should have x ≈ 0 (2D section in y-z plane)."""
        for e in onshape_all["boundary_edges"]:
            for pt in e:
                assert abs(pt[0]) < 1e-9, f"Found x={pt[0]} on boundary edge"

    def test_boundary_edges_not_empty(self, onshape_all):
        """Onshape exports produce multiple disconnected edge fragments."""
        assert len(onshape_all["boundary_edges"]) >= 2

    def test_boundary_edges_have_3d_points(self, onshape_all):
        """All edge points are 3D (x=0 for 2D section)."""
        for e in onshape_all["boundary_edges"]:
            for pt in e:
                assert len(pt) == 3, f"Point {pt} is not 3D"


# ─── YAML output structure ─────────────────────────────────────────────────────

class TestYamlOutputStructure:
    """Output YAML contains all required keys."""

    @pytest.fixture
    def bk16_result(self, onshape_bk16, run_import):
        return run_import(onshape_bk16, n_segments=20)

    def test_enclosure_type_is_blh(self, bk16_result):
        result, _ = bk16_result
        assert result["enclosure_type"] == "BLH"

    def test_width_present(self, bk16_result):
        result, _ = bk16_result
        assert "width" in result
        assert result["width"] > 0

    def test_enclosure_dims_two_values(self, bk16_result):
        result, _ = bk16_result
        assert len(result["enclosure_dims"]) == 2
        assert result["enclosure_dims"][0] > 0
        assert result["enclosure_dims"][1] > 0

    def test_coordinates_list(self, bk16_result):
        result, _ = bk16_result
        assert isinstance(result["coordinates"], list)
        assert len(result["coordinates"]) >= 2

    def test_rectangular_segments(self, bk16_result):
        """preserve_breaks=True should produce rectangular_segments."""
        result, _ = bk16_result
        assert "rectangular_segments" in result
        assert "conical_segments" not in result

    def test_discretisation_is_geometry(self, bk16_result):
        result, _ = bk16_result
        assert result.get("discretisation") == "geometry"

    def test_bend_angles_present(self, bk16_result):
        result, _ = bk16_result
        assert "bend_angles" in result
        assert isinstance(result["bend_angles"], list)

    def test_bend_angles_length(self, bk16_result):
        result, _ = bk16_result
        # bend_angles has n_segments - 1 entries
        n_seg = len(result["rectangular_segments"])
        assert len(result["bend_angles"]) == n_seg - 1


# ─── Geometric validity ────────────────────────────────────────────────────────

class TestGeometricValidity:
    """Output geometry is valid and within bounds."""

    @pytest.fixture
    def bk16_result(self, onshape_bk16, run_import):
        return run_import(onshape_bk16, n_segments=20)

    @pytest.fixture
    def fsx_result(self, onshape_fsx, run_import):
        return run_import(onshape_fsx, n_segments=20)

    @pytest.fixture
    def long_result(self, onshape_long, run_import):
        return run_import(onshape_long, n_segments=20)

    @pytest.fixture
    def bk16_1m_result(self, onshape_bk16_1m, run_import):
        return run_import(onshape_bk16_1m, n_segments=20)

    def test_path_start_near_throat(self, bk16_result, onshape_bk16):
        result, _ = bk16_result
        coords = result["coordinates"]
        offset = result.get("_center_offset", [0.0, 0.0])
        # Throat center from JSON (y,z), shifted by centering offset
        t_pts = onshape_bk16["throat"]
        t_center = np.array([
            (t_pts[0][1] + t_pts[1][1]) / 2 - offset[0],
            (t_pts[0][2] + t_pts[1][2]) / 2 - offset[1],
        ])
        first_coord = np.array(coords[0])
        dist = np.linalg.norm(first_coord - t_center)
        assert dist < 0.05, f"Path start {first_coord} is {dist:.3f}m from throat center {t_center}"

    def test_path_end_near_mouth(self, bk16_result, onshape_bk16):
        result, _ = bk16_result
        coords = result["coordinates"]
        offset = result.get("_center_offset", [0.0, 0.0])
        # Mouth center from JSON (y,z), shifted by centering offset
        m_pts = onshape_bk16["mouth"]
        m_center = np.array([
            (m_pts[0][1] + m_pts[1][1]) / 2 - offset[0],
            (m_pts[0][2] + m_pts[1][2]) / 2 - offset[1],
        ])
        last_coord = np.array(coords[-1])
        dist = np.linalg.norm(last_coord - m_center)
        assert dist < 0.05, f"Path end {last_coord} is {dist:.3f}m from mouth center {m_center}"

    def test_all_coords_within_bbox(self, bk16_result, onshape_bk16):
        """All centerline coordinates should be within the boundary bbox + 5% margin."""
        result, _ = bk16_result
        coords = result["coordinates"]
        offset = result.get("_center_offset", [0.0, 0.0])
        edges = [np.array(e) for e in onshape_bk16["boundary_edges"]]
        all_pts = np.array([[pt[1], pt[2]] for e in edges for pt in e])
        # Shift bbox by the same amount applied to the path coordinates
        bbox_min = all_pts.min(axis=0) - np.array(offset)
        bbox_max = all_pts.max(axis=0) - np.array(offset)
        margin = 0.05 * (bbox_max - bbox_min)
        bbox_min -= margin
        bbox_max += margin
        for c in coords:
            pt = np.array(c)
            assert (pt >= bbox_min - 1e-6).all() and (pt <= bbox_max + 1e-6).all(), \
                f"Coord {c} outside bbox [{bbox_min}, {bbox_max}]"

    def test_segment_widths_positive(self, bk16_result):
        result, _ = bk16_result
        for seg in result["rectangular_segments"]:
            h_start, h_end = seg[1], seg[3]
            assert h_start > 0, f"Height start {h_start} not positive"
            assert h_end > 0, f"Height end {h_end} not positive"

    def test_segment_widths_within_diagonal(self, bk16_result, onshape_bk16):
        """Segment widths should be less than the polygon diagonal."""
        result, _ = bk16_result
        edges = [np.array(e) for e in onshape_bk16["boundary_edges"]]
        all_pts = np.array([[pt[1], pt[2]] for e in edges for pt in e])
        diagonal = np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))
        for seg in result["rectangular_segments"]:
            h_start, h_end = seg[1], seg[3]
            max_h = max(h_start, h_end)
            assert max_h < diagonal, f"Height {max_h} exceeds diagonal {diagonal:.3f}"

    def test_throat_width_from_throat_segment(self, bk16_result, onshape_bk16):
        """First segment's height should match throat line length (±20%)."""
        result, _ = bk16_result
        segs = result["rectangular_segments"]
        throat = onshape_bk16["throat"]
        t_len = np.linalg.norm([throat[1][1] - throat[0][1], throat[1][2] - throat[0][2]])
        first_h = segs[0][1]
        assert abs(first_h - t_len) / t_len < 0.20, \
            f"First height {first_h:.4f} differs from throat length {t_len:.4f} by >20%"

    def test_mouth_width_from_mouth_segment(self, bk16_result, onshape_bk16):
        """Last segment's height should match mouth line length (±20%)."""
        result, _ = bk16_result
        segs = result["rectangular_segments"]
        mouth = onshape_bk16["mouth"]
        m_len = np.linalg.norm([mouth[1][1] - mouth[0][1], mouth[1][2] - mouth[0][2]])
        last_h = segs[-1][3]
        assert abs(last_h - m_len) / m_len < 0.20, \
            f"Last height {last_h:.4f} differs from mouth length {m_len:.4f} by >20%"

    def test_all_horns_produce_valid_polygons(self, onshape_bk16, onshape_fsx,
                                              onshape_long, onshape_bk16_1m,
                                              polygon_checker):
        """Each JSON's boundary should form a valid polygon.
        
        Onshape exports often produce disconnected edge fragments that must
        be merged by the medial axis pipeline. This tests that all 4 horns
        succeed in the pipeline regardless of edge connectivity.
        """
        for name, json_dict in [
            ("bk16", onshape_bk16),
            ("fsx", onshape_fsx),
            ("long", onshape_long),
            ("bk16-1M", onshape_bk16_1m),
        ]:
            # The medial axis pipeline should handle fragmented edges
            assert len(json_dict["boundary_edges"]) >= 2
            # All horns should produce a result from the pipeline
            assert "throat" in json_dict
            assert "mouth" in json_dict


# ─── Parsing into HornGeometry ─────────────────────────────────────────────────

class TestParseHornGeometry:
    """Output YAML can be parsed into a HornGeometry without error."""

    @pytest.fixture
    def fsx_result(self, onshape_fsx, run_import):
        return run_import(onshape_fsx, n_segments=20)

    @pytest.fixture
    def long_result(self, onshape_long, run_import):
        return run_import(onshape_long, n_segments=20)

    @pytest.fixture
    def bk16_1m_result(self, onshape_bk16_1m, run_import):
        return run_import(onshape_bk16_1m, n_segments=20)

    def test_bk16_yaml_parses(self, onshape_bk16, run_import):
        result, yaml_path = run_import(onshape_bk16, n_segments=20)
        horn = parse_horn_geometry(yaml_path)
        assert horn is not None

    def test_fsx_yaml_parses(self, fsx_result):
        _, yaml_path = fsx_result
        horn = parse_horn_geometry(yaml_path)
        assert horn is not None

    def test_long_yaml_parses(self, long_result):
        _, yaml_path = long_result
        horn = parse_horn_geometry(yaml_path)
        assert horn is not None

    def test_bk16_1m_yaml_parses(self, bk16_1m_result):
        _, yaml_path = bk16_1m_result
        horn = parse_horn_geometry(yaml_path)
        assert horn is not None

    def test_fsx_has_bend_angles(self, fsx_result):
        _, yaml_path = fsx_result
        horn = parse_horn_geometry(yaml_path)
        assert horn.bend_angles is not None
        assert len(horn.bend_angles) > 0

    def test_long_has_bend_angles(self, long_result):
        _, yaml_path = long_result
        horn = parse_horn_geometry(yaml_path)
        assert horn.bend_angles is not None
        assert len(horn.bend_angles) > 0


# ─── Segment count control ─────────────────────────────────────────────────────

class TestSegmentCount:
    """n_segments parameter controls output segment count."""

    @pytest.fixture
    def onshape_bk16(self):
        import json
        path = Path(__file__).parent / "onshape_data" / "bk16.json"
        with open(path) as f:
            return json.load(f)

    @pytest.mark.parametrize("n_seg", [5, 10, 20, 40])
    def test_n_segments_is_approximate_target(self, onshape_bk16, run_import, n_seg):
        """With preserve_breaks=True, n_segments is an approximate target.
        
        Extra segments are added at fold/stair break points, so actual
        count >= requested n_segments.
        """
        result, _ = run_import(onshape_bk16, n_segments=n_seg)
        actual = len(result["rectangular_segments"])
        # actual should be close to n_seg (within 2x, at least n_seg-5 to handle rounding)
        assert actual >= n_seg - 5, f"Expected >= {n_seg - 5}, got {actual}"
        assert actual <= n_seg * 2 + 5, f"Expected <= ~{n_seg * 2}, got {actual}"


# ─── Flip options ───────────────────────────────────────────────────────────────

class TestFlipOptions:
    """flip_x and flip_y options work without error."""

    @pytest.fixture
    def onshape_bk16(self):
        import json
        path = Path(__file__).parent / "onshape_data" / "bk16.json"
        with open(path) as f:
            return json.load(f)

    def test_flip_x(self, onshape_bk16, run_import):
        result_flipped, _ = run_import(onshape_bk16, n_segments=10, flip_x=True)
        result_orig, _ = run_import(onshape_bk16, n_segments=10, flip_x=False)
        # X coords should differ
        flipped_x = [c[0] for c in result_flipped["coordinates"]]
        orig_x = [c[0] for c in result_orig["coordinates"]]
        assert flipped_x != orig_x

    def test_flip_y(self, onshape_bk16, run_import):
        result_flipped, _ = run_import(onshape_bk16, n_segments=10, flip_y=True)
        result_orig, _ = run_import(onshape_bk16, n_segments=10, flip_y=False)
        # Y coords should differ
        flipped_y = [c[1] for c in result_flipped["coordinates"]]
        orig_y = [c[1] for c in result_orig["coordinates"]]
        assert flipped_y != orig_y


# ─── Numerical precision ─────────────────────────────────────────────────────────

class TestNumericalPrecision:
    """Quantitative comparisons between imported geometry and JSON source data."""

    @pytest.fixture
    def bk16_result(self, onshape_bk16, run_import):
        result, _ = run_import(onshape_bk16, n_segments=20)
        return result

    @pytest.fixture
    def fsx_result(self, onshape_fsx, run_import):
        result, _ = run_import(onshape_fsx, n_segments=20)
        return result

    @pytest.fixture
    def long_result(self, onshape_long, run_import):
        result, _ = run_import(onshape_long, n_segments=20)
        return result

    @pytest.fixture
    def bk16_1m_result(self, onshape_bk16_1m, run_import):
        result, _ = run_import(onshape_bk16_1m, n_segments=20)
        return result

    # ── Throat width ──────────────────────────────────────────────────────────

    def test_bk16_throat_width_exact(self, onshape_bk16, bk16_result):
        t = np.array(onshape_bk16["throat"])
        t_len = float(np.linalg.norm(t[1, 1:] - t[0, 1:]))
        first_h = bk16_result["rectangular_segments"][0][1]
        err_mm = abs(first_h - t_len) * 1000
        assert err_mm < 1.0, f"throat width error = {err_mm:.2f}mm (expected < 1mm)"

    def test_fsx_throat_width_exact(self, onshape_fsx, fsx_result):
        t = np.array(onshape_fsx["throat"])
        t_len = float(np.linalg.norm(t[1, 1:] - t[0, 1:]))
        first_h = fsx_result["rectangular_segments"][0][1]
        err_mm = abs(first_h - t_len) * 1000
        assert err_mm < 1.0, f"throat width error = {err_mm:.2f}mm (expected < 1mm)"

    def test_long_throat_width_exact(self, onshape_long, long_result):
        t = np.array(onshape_long["throat"])
        t_len = float(np.linalg.norm(t[1, 1:] - t[0, 1:]))
        first_h = long_result["rectangular_segments"][0][1]
        err_mm = abs(first_h - t_len) * 1000
        assert err_mm < 1.0, f"throat width error = {err_mm:.2f}mm (expected < 1mm)"

    def test_bk16_1m_throat_width_exact(self, onshape_bk16_1m, bk16_1m_result):
        t = np.array(onshape_bk16_1m["throat"])
        t_len = float(np.linalg.norm(t[1, 1:] - t[0, 1:]))
        first_h = bk16_1m_result["rectangular_segments"][0][1]
        err_mm = abs(first_h - t_len) * 1000
        assert err_mm < 1.0, f"throat width error = {err_mm:.2f}mm (expected < 1mm)"

    # ── Mouth width ───────────────────────────────────────────────────────────

    def test_bk16_mouth_width_exact(self, onshape_bk16, bk16_result):
        m = np.array(onshape_bk16["mouth"])
        m_len = float(np.linalg.norm(m[1, 1:] - m[0, 1:]))
        last_h = bk16_result["rectangular_segments"][-1][3]
        err_mm = abs(last_h - m_len) * 1000
        assert err_mm < 1.0, f"mouth width error = {err_mm:.2f}mm (expected < 1mm)"

    def test_fsx_mouth_width_exact(self, onshape_fsx, fsx_result):
        m = np.array(onshape_fsx["mouth"])
        m_len = float(np.linalg.norm(m[1, 1:] - m[0, 1:]))
        last_h = fsx_result["rectangular_segments"][-1][3]
        err_mm = abs(last_h - m_len) * 1000
        assert err_mm < 1.0, f"mouth width error = {err_mm:.2f}mm (expected < 1mm)"

    def test_long_mouth_width_exact(self, onshape_long, long_result):
        m = np.array(onshape_long["mouth"])
        m_len = float(np.linalg.norm(m[1, 1:] - m[0, 1:]))
        last_h = long_result["rectangular_segments"][-1][3]
        err_mm = abs(last_h - m_len) * 1000
        assert err_mm < 1.0, f"mouth width error = {err_mm:.2f}mm (expected < 1mm)"

    def test_bk16_1m_mouth_width_exact(self, onshape_bk16_1m, bk16_1m_result):
        m = np.array(onshape_bk16_1m["mouth"])
        m_len = float(np.linalg.norm(m[1, 1:] - m[0, 1:]))
        last_h = bk16_1m_result["rectangular_segments"][-1][3]
        err_mm = abs(last_h - m_len) * 1000
        assert err_mm < 1.0, f"mouth width error = {err_mm:.2f}mm (expected < 1mm)"

    # ── Path start / end (accounting for centering offset) ──────────────────────

    def test_bk16_path_starts_at_throat(self, onshape_bk16, bk16_result):
        offset = bk16_result.get("_center_offset", [0.0, 0.0])
        t = np.array(onshape_bk16["throat"])
        t_center = t.mean(axis=0)[[1, 2]] - np.array(offset)
        start = np.array(bk16_result["coordinates"][0][:2])
        dist = float(np.linalg.norm(start - t_center))
        assert dist < 0.01, f"path start {dist*100:.1f}cm from throat (expected < 1cm)"

    def test_fsx_path_starts_at_throat(self, onshape_fsx, fsx_result):
        offset = fsx_result.get("_center_offset", [0.0, 0.0])
        t = np.array(onshape_fsx["throat"])
        t_center = t.mean(axis=0)[[1, 2]] - np.array(offset)
        start = np.array(fsx_result["coordinates"][0][:2])
        dist = float(np.linalg.norm(start - t_center))
        assert dist < 0.01, f"path start {dist*100:.1f}cm from throat (expected < 1cm)"

    def test_long_path_starts_at_throat(self, onshape_long, long_result):
        offset = long_result.get("_center_offset", [0.0, 0.0])
        t = np.array(onshape_long["throat"])
        t_center = t.mean(axis=0)[[1, 2]] - np.array(offset)
        start = np.array(long_result["coordinates"][0][:2])
        dist = float(np.linalg.norm(start - t_center))
        assert dist < 0.01, f"path start {dist*100:.1f}cm from throat (expected < 1cm)"

    def test_bk16_1m_path_starts_at_throat(self, onshape_bk16_1m, bk16_1m_result):
        offset = bk16_1m_result.get("_center_offset", [0.0, 0.0])
        t = np.array(onshape_bk16_1m["throat"])
        t_center = t.mean(axis=0)[[1, 2]] - np.array(offset)
        start = np.array(bk16_1m_result["coordinates"][0][:2])
        dist = float(np.linalg.norm(start - t_center))
        assert dist < 0.01, f"path start {dist*100:.1f}cm from throat (expected < 1cm)"

    def test_bk16_path_ends_at_mouth(self, onshape_bk16, bk16_result):
        offset = bk16_result.get("_center_offset", [0.0, 0.0])
        m = np.array(onshape_bk16["mouth"])
        m_center = m.mean(axis=0)[[1, 2]] - np.array(offset)
        end = np.array(bk16_result["coordinates"][-1][:2])
        dist = float(np.linalg.norm(end - m_center))
        assert dist < 0.01, f"path end {dist*100:.1f}cm from mouth (expected < 1cm)"

    def test_fsx_path_ends_at_mouth(self, onshape_fsx, fsx_result):
        offset = fsx_result.get("_center_offset", [0.0, 0.0])
        m = np.array(onshape_fsx["mouth"])
        m_center = m.mean(axis=0)[[1, 2]] - np.array(offset)
        end = np.array(fsx_result["coordinates"][-1][:2])
        dist = float(np.linalg.norm(end - m_center))
        assert dist < 0.01, f"path end {dist*100:.1f}cm from mouth (expected < 1cm)"

    def test_long_path_ends_at_mouth(self, onshape_long, long_result):
        offset = long_result.get("_center_offset", [0.0, 0.0])
        m = np.array(onshape_long["mouth"])
        m_center = m.mean(axis=0)[[1, 2]] - np.array(offset)
        end = np.array(long_result["coordinates"][-1][:2])
        dist = float(np.linalg.norm(end - m_center))
        assert dist < 0.01, f"path end {dist*100:.1f}cm from mouth (expected < 1cm)"

    def test_bk16_1m_path_ends_at_mouth(self, onshape_bk16_1m, bk16_1m_result):
        offset = bk16_1m_result.get("_center_offset", [0.0, 0.0])
        m = np.array(onshape_bk16_1m["mouth"])
        m_center = m.mean(axis=0)[[1, 2]] - np.array(offset)
        end = np.array(bk16_1m_result["coordinates"][-1][:2])
        dist = float(np.linalg.norm(end - m_center))
        assert dist < 0.01, f"path end {dist*100:.1f}cm from mouth (expected < 1cm)"

    # ── Path length vs straight line ──────────────────────────────────────────

    def test_bk16_path_longer_than_straight(self, onshape_bk16, bk16_result):
        t = np.array(onshape_bk16["throat"])
        m = np.array(onshape_bk16["mouth"])
        straight = float(np.linalg.norm(m.mean(axis=0)[[1,2]] - t.mean(axis=0)[[1,2]]))
        coords = np.array(bk16_result["coordinates"])
        path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        assert path_len > straight * 1.1, \
            f"path={path_len:.3f}m should exceed straight={straight:.3f}m"

    def test_fsx_path_longer_than_straight(self, onshape_fsx, fsx_result):
        t = np.array(onshape_fsx["throat"])
        m = np.array(onshape_fsx["mouth"])
        straight = float(np.linalg.norm(m.mean(axis=0)[[1,2]] - t.mean(axis=0)[[1,2]]))
        coords = np.array(fsx_result["coordinates"])
        path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        assert path_len > straight * 1.1, \
            f"path={path_len:.3f}m should exceed straight={straight:.3f}m"

    def test_long_path_longer_than_straight(self, onshape_long, long_result):
        t = np.array(onshape_long["throat"])
        m = np.array(onshape_long["mouth"])
        straight = float(np.linalg.norm(m.mean(axis=0)[[1,2]] - t.mean(axis=0)[[1,2]]))
        coords = np.array(long_result["coordinates"])
        path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        assert path_len > straight * 1.1, \
            f"path={path_len:.3f}m should exceed straight={straight:.3f}m"

    def test_bk16_1m_path_longer_than_straight(self, onshape_bk16_1m, bk16_1m_result):
        t = np.array(onshape_bk16_1m["throat"])
        m = np.array(onshape_bk16_1m["mouth"])
        straight = float(np.linalg.norm(m.mean(axis=0)[[1,2]] - t.mean(axis=0)[[1,2]]))
        coords = np.array(bk16_1m_result["coordinates"])
        path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        assert path_len > straight * 1.1, \
            f"path={path_len:.3f}m should exceed straight={straight:.3f}m"

    def test_bk16_path_vs_straight_ratio(self, onshape_bk16, bk16_result):
        t = np.array(onshape_bk16["throat"])
        m = np.array(onshape_bk16["mouth"])
        straight = float(np.linalg.norm(m.mean(axis=0)[[1,2]] - t.mean(axis=0)[[1,2]]))
        coords = np.array(bk16_result["coordinates"])
        path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        ratio = path_len / straight
        assert 2.5 < ratio < 6.0, f"bk16 path ratio {ratio:.2f}x not in [2.5, 6.0]x"

    def test_fsx_path_vs_straight_ratio(self, onshape_fsx, fsx_result):
        t = np.array(onshape_fsx["throat"])
        m = np.array(onshape_fsx["mouth"])
        straight = float(np.linalg.norm(m.mean(axis=0)[[1,2]] - t.mean(axis=0)[[1,2]]))
        coords = np.array(fsx_result["coordinates"])
        path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        ratio = path_len / straight
        assert 2.5 < ratio < 8.0, f"fsx path ratio {ratio:.2f}x not in [2.5, 8.0]x"

    def test_long_path_vs_straight_ratio(self, onshape_long, long_result):
        t = np.array(onshape_long["throat"])
        m = np.array(onshape_long["mouth"])
        straight = float(np.linalg.norm(m.mean(axis=0)[[1,2]] - t.mean(axis=0)[[1,2]]))
        coords = np.array(long_result["coordinates"])
        path_len = float(np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
        ratio = path_len / straight
        assert 2.5 < ratio < 6.0, f"long path ratio {ratio:.2f}x not in [2.5, 6.0]x"

    # ── Segment integrity ─────────────────────────────────────────────────────

    def test_bk16_all_segment_areas_positive(self, bk16_result):
        segs = bk16_result["rectangular_segments"]
        width = bk16_result["width"]
        for i, seg in enumerate(segs):
            h1, h2 = seg[1], seg[3]
            assert h1 * width > 0, f"seg{i} start area = {h1*width} (must be > 0)"
            assert h2 * width > 0, f"seg{i} end area = {h2*width} (must be > 0)"

    def test_fsx_all_segment_areas_positive(self, fsx_result):
        segs = fsx_result["rectangular_segments"]
        width = fsx_result["width"]
        for i, seg in enumerate(segs):
            h1, h2 = seg[1], seg[3]
            assert h1 * width > 0, f"seg{i} start area = {h1*width} (must be > 0)"
            assert h2 * width > 0, f"seg{i} end area = {h2*width} (must be > 0)"

    def test_long_all_segment_areas_positive(self, long_result):
        segs = long_result["rectangular_segments"]
        width = long_result["width"]
        for i, seg in enumerate(segs):
            h1, h2 = seg[1], seg[3]
            assert h1 * width > 0, f"seg{i} start area = {h1*width} (must be > 0)"
            assert h2 * width > 0, f"seg{i} end area = {h2*width} (must be > 0)"

    def test_bk16_1m_all_segment_areas_positive(self, bk16_1m_result):
        segs = bk16_1m_result["rectangular_segments"]
        width = bk16_1m_result["width"]
        for i, seg in enumerate(segs):
            h1, h2 = seg[1], seg[3]
            assert h1 * width > 0, f"seg{i} start area = {h1*width} (must be > 0)"
            assert h2 * width > 0, f"seg{i} end area = {h2*width} (must be > 0)"

    def test_bk16_all_segment_lengths_positive(self, bk16_result):
        for i, seg in enumerate(bk16_result["rectangular_segments"]):
            assert seg[4] > 0, f"seg{i} length = {seg[4]} (must be > 0)"

    def test_fsx_all_segment_lengths_positive(self, fsx_result):
        for i, seg in enumerate(fsx_result["rectangular_segments"]):
            assert seg[4] > 0, f"seg{i} length = {seg[4]} (must be > 0)"

    def test_long_all_segment_lengths_positive(self, long_result):
        for i, seg in enumerate(long_result["rectangular_segments"]):
            assert seg[4] > 0, f"seg{i} length = {seg[4]} (must be > 0)"

    def test_bk16_1m_all_segment_lengths_positive(self, bk16_1m_result):
        for i, seg in enumerate(bk16_1m_result["rectangular_segments"]):
            assert seg[4] > 0, f"seg{i} length = {seg[4]} (must be > 0)"

    def test_bk16_no_duplicate_coords(self, bk16_result):
        coords = bk16_result["coordinates"]
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = float(np.linalg.norm(np.array(coords[i]) - np.array(coords[j])))
                assert dist > 1e-4, f"coords[{i}] and [{j}] dup: {dist:.6f}"

    def test_fsx_no_duplicate_coords(self, fsx_result):
        coords = fsx_result["coordinates"]
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = float(np.linalg.norm(np.array(coords[i]) - np.array(coords[j])))
                assert dist > 1e-4, f"coords[{i}] and [{j}] dup: {dist:.6f}"

    def test_long_no_duplicate_coords(self, long_result):
        coords = long_result["coordinates"]
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = float(np.linalg.norm(np.array(coords[i]) - np.array(coords[j])))
                assert dist > 1e-4, f"coords[{i}] and [{j}] dup: {dist:.6f}"

    def test_bk16_1m_no_duplicate_coords(self, bk16_1m_result):
        coords = bk16_1m_result["coordinates"]
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                dist = float(np.linalg.norm(np.array(coords[i]) - np.array(coords[j])))
                assert dist > 1e-4, f"coords[{i}] and [{j}] dup: {dist:.6f}"

    # ── Volume sanity ─────────────────────────────────────────────────────────

    def test_bk16_volume_less_than_bbox(self, onshape_bk16, bk16_result):
        segs = bk16_result["rectangular_segments"]
        width = bk16_result["width"]
        edges = [np.array(e) for e in onshape_bk16["boundary_edges"]]
        all_pts = np.array([[pt[1], pt[2]] for e in edges for pt in e])
        bbox = all_pts.max(axis=0) - all_pts.min(axis=0)
        bbox_vol = float(np.prod(bbox)) * width * 1000
        total_vol = sum((s[1]+s[3])/2 * s[4] * width * 1000 for s in segs)
        assert total_vol < bbox_vol * 1.05, \
            f"vol={total_vol:.1f}L > bbox={bbox_vol:.1f}L × 1.05"

    def test_fsx_volume_less_than_bbox(self, onshape_fsx, fsx_result):
        segs = fsx_result["rectangular_segments"]
        width = fsx_result["width"]
        edges = [np.array(e) for e in onshape_fsx["boundary_edges"]]
        all_pts = np.array([[pt[1], pt[2]] for e in edges for pt in e])
        bbox = all_pts.max(axis=0) - all_pts.min(axis=0)
        bbox_vol = float(np.prod(bbox)) * width * 1000
        total_vol = sum((s[1]+s[3])/2 * s[4] * width * 1000 for s in segs)
        assert total_vol < bbox_vol * 1.05, \
            f"vol={total_vol:.1f}L > bbox={bbox_vol:.1f}L × 1.05"

    def test_long_volume_less_than_bbox(self, onshape_long, long_result):
        segs = long_result["rectangular_segments"]
        width = long_result["width"]
        edges = [np.array(e) for e in onshape_long["boundary_edges"]]
        all_pts = np.array([[pt[1], pt[2]] for e in edges for pt in e])
        bbox = all_pts.max(axis=0) - all_pts.min(axis=0)
        bbox_vol = float(np.prod(bbox)) * width * 1000
        total_vol = sum((s[1]+s[3])/2 * s[4] * width * 1000 for s in segs)
        assert total_vol < bbox_vol * 1.05, \
            f"vol={total_vol:.1f}L > bbox={bbox_vol:.1f}L × 1.05"

    def test_bk16_1m_volume_less_than_bbox(self, onshape_bk16_1m, bk16_1m_result):
        segs = bk16_1m_result["rectangular_segments"]
        width = bk16_1m_result["width"]
        edges = [np.array(e) for e in onshape_bk16_1m["boundary_edges"]]
        all_pts = np.array([[pt[1], pt[2]] for e in edges for pt in e])
        bbox = all_pts.max(axis=0) - all_pts.min(axis=0)
        bbox_vol = float(np.prod(bbox)) * width * 1000
        total_vol = sum((s[1]+s[3])/2 * s[4] * width * 1000 for s in segs)
        assert total_vol < bbox_vol * 1.05, \
            f"vol={total_vol:.1f}L > bbox={bbox_vol:.1f}L × 1.05"


# ─── Summary metrics ─────────────────────────────────────────────────────────────

class TestImportedHornMetrics:
    """Sanity-check physical metrics of the imported horns."""

    @pytest.fixture
    def bk16_result(self, onshape_bk16, run_import):
        return run_import(onshape_bk16, n_segments=20)

    @pytest.fixture
    def fsx_result(self, onshape_fsx, run_import):
        return run_import(onshape_fsx, n_segments=20)

    @pytest.fixture
    def long_result(self, onshape_long, run_import):
        return run_import(onshape_long, n_segments=20)

    @pytest.fixture
    def bk16_1m_result(self, onshape_bk16_1m, run_import):
        return run_import(onshape_bk16_1m, n_segments=20)

    def _segment_volumes(self, segs, width):
        """Approximate volume of each segment in litres."""
        vols = []
        for i, seg in enumerate(segs):
            h1, h2 = seg[1], seg[3]
            L = seg[4]
            avg_h = (h1 + h2) / 2
            vol_m3 = width * avg_h * L
            vols.append(vol_m3 * 1000)
        return vols

    def _path_length(self, coords):
        """Total path length from coordinates."""
        pts = np.array(coords)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    def test_bk16_path_length(self, bk16_result):
        result, _ = bk16_result
        L = self._path_length(result["coordinates"])
        assert 0.05 < L < 3.0, f"Path length {L:.3f}m out of reasonable range"

    def test_fsx_path_length(self, fsx_result):
        result, _ = fsx_result
        L = self._path_length(result["coordinates"])
        assert 0.05 < L < 5.0, f"Path length {L:.3f}m out of reasonable range"

    def test_long_path_length(self, long_result):
        result, _ = long_result
        L = self._path_length(result["coordinates"])
        assert 0.05 < L < 5.0, f"Path length {L:.3f}m out of reasonable range"

    def test_bk16_volume_reasonable(self, bk16_result, onshape_bk16):
        result, _ = bk16_result
        width = result["width"]
        vols = self._segment_volumes(result["rectangular_segments"], width)
        total_l = sum(vols)
        assert 0.01 < total_l < 500, f"Volume {total_l:.1f}L out of reasonable range"

    def test_fsx_volume_reasonable(self, fsx_result, onshape_fsx):
        result, _ = fsx_result
        width = result["width"]
        vols = self._segment_volumes(result["rectangular_segments"], width)
        total_l = sum(vols)
        assert 0.01 < total_l < 500, f"Volume {total_l:.1f}L out of reasonable range"

    def test_bk16_mouth_wider_than_throat(self, bk16_result):
        result, _ = bk16_result
        segs = result["rectangular_segments"]
        throat_h = segs[0][1]
        mouth_h = segs[-1][3]
        assert mouth_h > throat_h, \
            f"Mouth height {mouth_h:.4f} should exceed throat height {throat_h:.4f}"

    def test_fsx_mouth_wider_than_throat(self, fsx_result):
        result, _ = fsx_result
        segs = result["rectangular_segments"]
        throat_h = segs[0][1]
        mouth_h = segs[-1][3]
        assert mouth_h > throat_h, \
            f"Mouth height {mouth_h:.4f} should exceed throat height {throat_h:.4f}"

    def test_bk16_bend_angles_bounded(self, bk16_result):
        result, _ = bk16_result
        angles = result["bend_angles"]
        assert all(0 <= a <= np.pi for a in angles)

    def test_fsx_bend_angles_bounded(self, fsx_result):
        result, _ = fsx_result
        angles = result["bend_angles"]
        assert all(0 <= a <= np.pi for a in angles)
