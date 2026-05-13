"""Pytest fixtures for Onshape JSON import tests and Hornresp comparison framework."""

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import numpy as np


# ─── Raw JSON fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def onshape_bk16():
    path = Path(__file__).parent / "onshape_data" / "bk16.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def onshape_fsx():
    path = Path(__file__).parent / "onshape_data" / "fsx.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def onshape_long():
    path = Path(__file__).parent / "onshape_data" / "long.json"
    with open(path) as f:
        return json.load(f)


@pytest.fixture
def onshape_bk16_1m():
    path = Path(__file__).parent / "onshape_data" / "bk16-1M.json"
    with open(path) as f:
        return json.load(f)


# ─── All raw JSONs ──────────────────────────────────────────────────────────────

@pytest.fixture(params=["bk16", "fsx", "long", "bk16-1M"])
def onshape_all(request, onshape_bk16, onshape_fsx, onshape_long, onshape_bk16_1m):
    """Parametrized fixture cycling through all 4 JSON files."""
    return {"bk16": onshape_bk16, "fsx": onshape_fsx, "long": onshape_long, "bk16-1M": onshape_bk16_1m}[request.param]


# ─── Import runner ─────────────────────────────────────────────────────────────

@pytest.fixture
def run_import(tmp_path):
    """
    Run generate_auto_segments with geometry_aware + preserve_breaks on a JSON dict.
    Returns (result_dict, output_yaml_path).
    Coordinates are normalised to lists of lists for test compatibility.
    """
    from pyhorn_core.solver.medial_axis import generate_auto_segments

    def _run(json_dict, n_segments=20, flip_x=False, flip_y=False, output_format="legacy"):
        json_path = tmp_path / "input.json"
        output_yaml = tmp_path / "output.yaml"
        with open(json_path, "w") as f:
            json.dump(json_dict, f)
        result = generate_auto_segments(
            json_path,
            output_yaml,
            n_segments=n_segments,
            flip_x=flip_x,
            flip_y=flip_y,
            geometry_aware=True,
            preserve_breaks=True,
            output_format=output_format,
        )
        # Patch the output YAML to include throat_area, mouth_area, path_length
        # which are required by parse_horn_geometry but not emitted by the
        # generate_auto_segments legacy format.
        import yaml
        with open(output_yaml) as f:
            yaml_data = yaml.safe_load(f) or {}
        # Compute throat_area, mouth_area from width * first/last segment heights
        width = yaml_data.get("width", result.get("width", 0.2))
        rectangular = result.get("rectangular_segments", [])
        if rectangular:
            throat_h = rectangular[0][1]  # h_start of first segment
            mouth_h = rectangular[-1][3]   # h_end of last segment
            yaml_data["throat_area"] = width * throat_h
            yaml_data["mouth_area"] = width * mouth_h
            yaml_data["path_length"] = sum(s[4] for s in rectangular)
        # Re-write so parse_horn_geometry sees the complete YAML
        with open(output_yaml, "w") as f:
            yaml.safe_dump(yaml_data, f, default_flow_style=None, sort_keys=False)
        # Normalise nested structures to lists (YAML may return tuples)
        def _to_list(x):
            if isinstance(x, (list, tuple)):
                return [float(v) if isinstance(v, (int, float)) else _to_list(v) for v in x]
            return x
        result["coordinates"] = [_to_list(c) for c in result["coordinates"]]
        result["rectangular_segments"] = [_to_list(s) for s in result.get("rectangular_segments", [])]
        if "bend_angles" in result and isinstance(result["bend_angles"], (list, tuple)):
            result["bend_angles"] = [_to_list(a) for a in result["bend_angles"]]
        return result, output_yaml

    return _run


# ─── Coordinate helpers ───────────────────────────────────────────────────────

def _json_center(json_dict, key):
    """Return (y, z) center of a throat/mouth from a JSON dict's 3D points."""
    pts = json_dict[key]  # [[x,y,z], [x,y,z]]
    ys = [pts[0][1], pts[1][1]]
    zs = [pts[0][2], pts[1][2]]
    return np.array([(ys[0] + ys[1]) / 2, (zs[0] + zs[1]) / 2])


# ─── Geometry helpers ──────────────────────────────────────────────────────────

@pytest.fixture
def polygon_checker():
    """Check that boundary_edges form a valid closed polygon."""
    from shapely.geometry import Polygon, LineString, MultiLineString
    from shapely.ops import linemerge, polygonize, unary_union
    import numpy as np

    def check(json_dict):
        edges = [np.array(e) for e in json_dict["boundary_edges"]]
        lines = [LineString(e[:, 1:]) for e in edges if len(e) >= 2]
        merged = linemerge(lines)
        polys = list(polygonize(merged))
        if not polys:
            noded = unary_union(lines)
            polys = list(polygonize(noded))
        if not polys:
            return False, "Could not form polygon from boundary_edges"
        poly = max(polys, key=lambda p: p.area)
        return poly.is_valid, f"Polygon area={poly.area:.6f}, valid={poly.is_valid}"

    return check


# ─── Auto-segment fixtures ────────────────────────────────────────────────────

@pytest.fixture
def valid_2d_json():
    """Minimal 2D polygon JSON for auto-segment tests."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Hornresp comparison framework
# ─────────────────────────────────────────────────────────────────────────────

import csv


@dataclass(frozen=True)
class HornrespReference:
    """Frequency-by-frequency reference values from Hornresp.

    Export from Hornresp: File → Export response data → CSV
    Columns: frequency, spl, re_z, im_z
    """

    freq: np.ndarray
    spl: np.ndarray
    re_z: np.ndarray
    im_z: np.ndarray

    @classmethod
    def from_csv(cls, path: str | Path) -> "HornrespReference":
        freq, spl, re_z, im_z = [], [], [], []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                freq.append(float(row["frequency"]))
                spl.append(float(row["spl"]))
                re_z.append(float(row["re_z"]))
                im_z.append(float(row["im_z"]))
        return cls(freq=np.array(freq), spl=np.array(spl),
                   re_z=np.array(re_z), im_z=np.array(im_z))


@dataclass
class ConvergenceResult:
    metric: str
    max_error: float
    mean_error: float
    rms_error: float
    worst_freq: float
    passed: bool


def _converge(
    py_vals, ref_vals, ref_freq, py_freq, metric, atol
) -> ConvergenceResult:
    aligned = np.interp(py_freq, ref_freq, ref_vals, left=np.nan, right=np.nan)
    valid = ~np.isnan(aligned)
    if not valid.any():
        return ConvergenceResult(metric, np.inf, np.inf, np.inf, 0.0, False)
    err = aligned[valid] - py_vals[valid]
    worst_idx = int(np.argmax(np.abs(err)))
    return ConvergenceResult(
        metric=metric,
        max_error=float(np.max(np.abs(err))),
        mean_error=float(np.mean(err)),
        rms_error=float(np.sqrt(np.mean(err**2))),
        worst_freq=float(py_freq[valid][worst_idx]),
        passed=float(np.max(np.abs(err)) <= atol),
    )


def assert_spl_convergence(pyhorn_result, ref: HornrespReference, atol=1.0):
    py_freq, py_spl = pyhorn_result.freqs, pyhorn_result.spl
    result = _converge(py_spl, ref.spl, ref.freq, py_freq, "SPL", atol)
    if not result.passed:
        pytest.fail(
            f"SPL divergence > {atol} dB | "
            f"max={result.max_error:.2f} dB at {result.worst_freq:.1f} Hz, "
            f"rms={result.rms_error:.2f} dB"
        )


def assert_impedance_convergence(pyhorn_result, ref: HornrespReference, atol=1.0):
    py_z = np.abs(pyhorn_result.impedance)
    py_freq = pyhorn_result.freqs
    ref_z = np.sqrt(ref.re_z**2 + ref.im_z**2)
    result = _converge(py_z, ref_z, ref.freq, py_freq, "|Z|", atol)
    if not result.passed:
        pytest.fail(
            f"|Z| divergence > {atol} Ω | "
            f"max={result.max_error:.2f} Ω at {result.worst_freq:.1f} Hz"
        )
