"""Regression test: re-run simulation for known geometries and compare to golden SPL.

This test validates that pyhorn's simulation produces consistent results for key
reference geometries. It loads stored "golden" CSV files (generated on 2026-05-03
when the geometries were confirmed stable), re-runs the simulation with the
same driver + geometry, and asserts the SPL difference is below threshold.

If this test fails, it means the solver has regressed and the golden files
need to be regenerated (or the threshold adjusted if intentional changes were made).

Run:
    pytest pyhorn_core/tests/test_golden_regression.py -v

Regenerate golden files (after intentional solver changes):
    python pyhorn_core/tests/test_golden_regression.py --regenerate
"""

from __future__ import annotations

import csv
import argparse
from pathlib import Path

import numpy as np
import pytest

from pyhorn_core.config.parser import (
    parse_driver_specs,
    parse_horn_geometry,
    parse_horn_project,
)
from pyhorn_core.solver.models import horn_response


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

TESTS_DIR = Path(__file__).parent
GOLDEN_DIR = TESTS_DIR / "golden"
SOURCE_DIR = Path(__file__).parent.parent.parent / "source"
PROJECTS_DIR = Path(__file__).parent.parent.parent / "projects"
DRIVER_YAML = Path(__file__).parent.parent.parent / "drivers" / "FE166NV2.yaml"

# Geometry name → (golden CSV filename, geometry YAML path)
# hiro: uses project YAML (has rear chamber params)
# fsx: uses geometry YAML (no rear chamber)
# bk16: uses project YAML (no rear chamber but vrc=0)
GOLDEN_GEOMETRIES = {
    "hiro": ("hiro_response.csv", PROJECTS_DIR / "hiro.yaml",   "project"),
    "fsx":  ("fsx_response.csv",  SOURCE_DIR / "fsx.yaml",      "geometry"),
    "bk16": ("bk16_response.csv", PROJECTS_DIR / "bk16.yaml",   "project"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_golden(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (frequency, SPL) arrays from a golden CSV file."""
    freq, spl = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq.append(float(row["Frequency_Hz"]))
            spl.append(float(row["SPL_dB_Horn SPL (dB)"]))
    return np.array(freq), np.array(spl)


def _load_horn(geo_yaml_path: Path, source: str):
    """Load HornGeometry from YAML. 'source' determines parser."""
    if source == "project":
        _, horn = parse_horn_project(geo_yaml_path)
    else:
        horn = parse_horn_geometry(geo_yaml_path)
    return horn


def _run_simulation(geo_yaml_path: Path, source: str, freqs: np.ndarray) -> np.ndarray:
    """Run horn_response for a geometry. Returns SPL array."""
    assert geo_yaml_path.exists(), f"YAML not found: {geo_yaml_path}"
    driver = parse_driver_specs(DRIVER_YAML)
    horn = _load_horn(geo_yaml_path, source)
    result = horn_response(
        freqs=freqs,
        driver=driver,
        horn=horn,
        compute_distortion=False,
    )
    return result.spl


def _spl_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    """Compute statistics for the SPL difference between two arrays."""
    diff = a - b
    return {
        "max_abs": float(np.max(np.abs(diff))),
        "rms":     float(np.sqrt(np.mean(diff**2))),
        "mean":    float(np.mean(diff)),
        "std":     float(np.std(diff)),
    }


def _write_diff_csv(path: Path, freqs: np.ndarray,
                    golden_spl: np.ndarray, current_spl: np.ndarray,
                    threshold: float) -> dict:
    """Write a per-frequency SPL diff CSV and return summary stats.

    CSV columns: Frequency_Hz, Golden_SPL_dB, Current_SPL_dB, Diff_dB, Abs_Diff_dB, Status
    Status = PASS if |diff| <= threshold, FAIL otherwise.
    Returns summary dict with pass/fail counts and worst-frequency info.
    """
    diff = current_spl - golden_spl
    abs_diff = np.abs(diff)
    status = np.where(abs_diff <= threshold, "PASS", "FAIL")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Frequency_Hz", "Golden_SPL_dB", "Current_SPL_dB",
            "Diff_dB", "Abs_Diff_dB", "Status",
        ])
        writer.writeheader()
        for i in range(len(freqs)):
            writer.writerow({
                "Frequency_Hz":   f"{freqs[i]:.4f}",
                "Golden_SPL_dB":  f"{golden_spl[i]:.4f}",
                "Current_SPL_dB": f"{current_spl[i]:.4f}",
                "Diff_dB":        f"{diff[i]:.4f}",
                "Abs_Diff_dB":    f"{abs_diff[i]:.4f}",
                "Status":         status[i],
            })

    fail_mask = status == "FAIL"
    worst_idx = int(np.argmax(abs_diff))
    stats = _spl_diff_stats(golden_spl, current_spl)
    return {
        "total":      len(freqs),
        "passed":     int(np.sum(~fail_mask)),
        "failed":     int(np.sum(fail_mask)),
        "max_abs":    stats["max_abs"],
        "rms":        stats["rms"],
        "mean":       stats["mean"],
        "std":        stats["std"],
        "worst_freq":  float(freqs[worst_idx]),
        "worst_diff":  float(abs_diff[worst_idx]),
        "worst_golden": float(golden_spl[worst_idx]),
        "worst_current": float(current_spl[worst_idx]),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(params=list(GOLDEN_GEOMETRIES.items()))
def geometry_name_and_paths(request):
    name, (csv_name, yaml_path, source) = request.param
    golden_csv = GOLDEN_DIR / csv_name
    return name, golden_csv, yaml_path, source


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGoldenRegression:
    """Regression tests against stored golden SPL files."""

    def test_golden_spl_regression(self, geometry_name_and_paths, request):
        """Re-run simulation and assert SPL diff vs golden < 0.5 dB."""
        name, golden_csv, geo_yaml, source = geometry_name_and_paths

        # Check files exist
        if not golden_csv.exists():
            pytest.skip(f"Golden CSV not found: {golden_csv}")
        if not geo_yaml.exists():
            pytest.skip(f"Geometry YAML not found: {geo_yaml}")
        if not DRIVER_YAML.exists():
            pytest.skip(f"Driver YAML not found: {DRIVER_YAML}")

        # Load golden
        golden_freq, golden_spl = _load_golden(golden_csv)

        # Re-run simulation
        current_spl = _run_simulation(geo_yaml, source, golden_freq)

        # Compare
        stats = _spl_diff_stats(golden_spl, current_spl)
        threshold = 0.5  # dB

        request.node.stash["spl_stats"] = stats

        assert stats["max_abs"] <= threshold, (
            f"[{name}] SPL regression detected: "
            f"max_abs={stats['max_abs']:.4f} dB > {threshold} dB  "
            f"(rms={stats['rms']:.4f}, mean={stats['mean']:+.4f}, std={stats['std']:.4f})"
        )

    def test_golden_spl_no_nan(self, geometry_name_and_paths):
        """Ensure simulation produces no NaN values."""
        name, golden_csv, geo_yaml, source = geometry_name_and_paths
        if not golden_csv.exists():
            pytest.skip(f"Golden CSV not found: {golden_csv}")

        golden_freq, _ = _load_golden(golden_csv)
        current_spl = _run_simulation(geo_yaml, source, golden_freq)

        nan_count = int(np.sum(np.isnan(current_spl)))
        assert nan_count == 0, f"[{name}] {nan_count} NaN values in SPL output"

    def test_golden_spl_sanity(self, geometry_name_and_paths):
        """Ensure SPL values are in a physically plausible range."""
        name, golden_csv, geo_yaml, source = geometry_name_and_paths
        if not golden_csv.exists():
            pytest.skip(f"Golden CSV not found: {golden_csv}")

        golden_freq, golden_spl = _load_golden(golden_csv)
        current_spl = _run_simulation(geo_yaml, source, golden_freq)

        # Physical sanity: SPL should be between 0 and 130 dB across the band
        nan_or_out_of_range = np.sum(np.isnan(current_spl) | (current_spl < 0) | (current_spl > 130))
        assert nan_or_out_of_range == 0, (
            f"[{name}] {nan_or_out_of_range} points out of physical range [0, 130] dB"
        )

    def test_golden_threshold_report(self, geometry_name_and_paths, tmp_path):
        """Generate a per-frequency diff CSV and assert threshold < 0.5 dB.

        Always writes a structured diff CSV (to tmp_path) regardless of pass/fail.
        The CSV makes it easy to inspect which frequencies are drifting and by how much.

        The test FAILS (regression) if max |diff| > 0.5 dB — the diff CSV is preserved
        as a build artifact for diagnosis.
        """
        name, golden_csv, geo_yaml, source = geometry_name_and_paths
        threshold = 0.5  # dB

        for path_ in [golden_csv, geo_yaml, DRIVER_YAML]:
            if not path_.exists():
                pytest.skip(f"Required file not found: {path_}")

        golden_freq, golden_spl = _load_golden(golden_csv)
        current_spl = _run_simulation(geo_yaml, source, golden_freq)

        # Write diff CSV to tmp_path (preserved as CI artifact on failure)
        diff_csv = tmp_path / f"spl_diff_{name}.csv"
        summary = _write_diff_csv(diff_csv, golden_freq, golden_spl, current_spl, threshold)

        # Assertion: max |diff| must be within threshold
        assert summary["max_abs"] <= threshold, (
            f"[{name}] SPL regression detected — "
            f"max_abs={summary['max_abs']:.4f} dB > {threshold} dB  "
            f"(rms={summary['rms']:.4f} dB, n_failed={summary['failed']}/{summary['total']})\n"
            f"  Worst at {summary['worst_freq']:.2f} Hz: "
            f"golden={summary['worst_golden']:.2f} dB, "
            f"current={summary['worst_current']:.2f} dB, "
            f"diff={summary['worst_diff']:.4f} dB\n"
            f"  Diff CSV: {diff_csv}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main — for regenerating golden files
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Regenerate golden SPL files")
    parser.add_argument("--regenerate", action="store_true", help="Overwrite golden CSV files")
    args = parser.parse_args()

    if not args.regenerate:
        print("Dry-run: use --regenerate to overwrite golden files")
        for name, (csv_name, yaml_path, source) in GOLDEN_GEOMETRIES.items():
            print(f"  {name}: {GOLDEN_DIR / csv_name}  (from {source} {yaml_path})")
        return

    print(f"Regenerating golden files in {GOLDEN_DIR} ...")
    driver = parse_driver_specs(DRIVER_YAML)

    for name, (csv_name, yaml_path, source) in GOLDEN_GEOMETRIES.items():
        if not yaml_path.exists():
            print(f"  SKIP {name}: YAML not found at {yaml_path}")
            continue

        horn = _load_horn(yaml_path, source)

        # Use a standard frequency grid: 20 Hz – 5000 Hz, 500 points
        fmin, fmax, n_points = 20.0, 5000.0, 500
        freqs = np.logspace(np.log10(fmin), np.log10(fmax), n_points)

        result = horn_response(
            freqs=freqs,
            driver=driver,
            horn=horn,
            compute_distortion=False,
        )

        out_path = GOLDEN_DIR / csv_name
        _write_golden_csv(out_path, result)
        print(
            f"  Wrote {out_path}  "
            f"({n_points} pts, SPL {result.spl.min():.1f}–{result.spl.max():.1f} dB)"
        )


def _write_golden_csv(path: Path, result) -> None:
    """Write a golden CSV file from a SimulationResult."""
    fieldnames = [
        "Frequency_Hz",
        "SPL_dB_Horn SPL (dB)",
        "SPL_dB_Impedance (Ohms)",
        "SPL_dB_Impedance Real (Ohms)",
        "SPL_dB_Impedance Imag (Ohms)",
        "SPL_dB_Impedance Phase (deg)",
        "SPL_dB_Excursion (mm)",
        "SPL_dB_Cone Velocity (m/s)",
        "SPL_dB_Phase (degrees)",
        "SPL_dB_Group delay (ms)",
        "SPL_dB_TMM artifact flag",
        "SPL_dB_Horn component SPL (dB)",
        "SPL_dB_Direct radiator SPL (dB)",
        "SPL_dB_Efficiency (%)",
        "SPL_dB_Driver Power (W)",
        "SPL_dB_Diaphragm Pressure Total (Pa)",
        "SPL_dB_Diaphragm Pressure Horn Side (Pa)",
        "SPL_dB_Diaphragm Pressure Direct Side (Pa)",
        "SPL_dB_Particle Velocity Throat (m/s)",
        "SPL_dB_Particle Velocity Mouth (m/s)",
        "SPL_dB_Particle Velocity Port (m/s)",
        "SPL_dB_DI 0° (dB)",
        "SPL_dB_DI 15° (dB)",
        "SPL_dB_DI 30° (dB)",
        "SPL_dB_DI 45° (dB)",
        "SPL_dB_DI 60° (dB)",
        "SPL_dB_DI 75° (dB)",
        "SPL_dB_DI 90° (dB)",
    ]

    def _col(key: str) -> np.ndarray:
        di = np.array(result.direction_index) if result.direction_index is not None else np.zeros((len(result.freqs), 7))
        di_vals = [di[:, i] if di.shape[1] > i else np.full(len(result.freqs), np.nan) for i in range(7)]
        # Convert numerical_artifacts list to float array (0.0 = no artifact, 1.0 = artifact)
        # numerical_artifacts is a list of frequency values where artifacts occur
        artifact_arr = np.zeros(len(result.freqs))
        for art_freq in result.numerical_artifacts:
            idx = np.argmin(np.abs(result.freqs - art_freq))
            if abs(result.freqs[idx] - art_freq) < 1.0:
                artifact_arr[idx] = 1.0
        return {
            "Frequency_Hz": result.freqs,
            "SPL_dB_Horn SPL (dB)": result.spl,
            "SPL_dB_Impedance (Ohms)": np.abs(result.impedance),
            "SPL_dB_Impedance Real (Ohms)": result.impedance.real,
            "SPL_dB_Impedance Imag (Ohms)": result.impedance.imag,
            "SPL_dB_Impedance Phase (deg)": np.angle(result.impedance, deg=True),
            "SPL_dB_Excursion (mm)": result.excursion,
            "SPL_dB_Cone Velocity (m/s)": result.cone_velocity,
            "SPL_dB_Phase (degrees)": np.rad2deg(result.phase),
            "SPL_dB_Group delay (ms)": result.group_delay * 1000.0,
            "SPL_dB_TMM artifact flag": artifact_arr,
            "SPL_dB_Horn component SPL (dB)": result.horn_spl,
            "SPL_dB_Direct radiator SPL (dB)": result.direct_spl,
            "SPL_dB_Efficiency (%)": result.efficiency_pct,
            "SPL_dB_Driver Power (W)": result.electrical_input_power,
            "SPL_dB_Diaphragm Pressure Total (Pa)": np.abs(result.diaphragm_pressure_total),
            "SPL_dB_Diaphragm Pressure Horn Side (Pa)": np.abs(result.diaphragm_pressure_horn_side),
            "SPL_dB_Diaphragm Pressure Direct Side (Pa)": np.abs(result.diaphragm_pressure_direct_side),
            "SPL_dB_Particle Velocity Throat (m/s)": result.particle_velocity_throat,
            "SPL_dB_Particle Velocity Mouth (m/s)": result.particle_velocity_mouth,
            "SPL_dB_Particle Velocity Port (m/s)": result.particle_velocity_port,
            "SPL_dB_DI 0° (dB)": di_vals[0],
            "SPL_dB_DI 15° (dB)": di_vals[1],
            "SPL_dB_DI 30° (dB)": di_vals[2],
            "SPL_dB_DI 45° (dB)": di_vals[3],
            "SPL_dB_DI 60° (dB)": di_vals[4],
            "SPL_dB_DI 75° (dB)": di_vals[5],
            "SPL_dB_DI 90° (dB)": di_vals[6],
        }.get(key, np.full_like(result.freqs, np.nan, dtype=float))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(result.freqs)):
            row = {name: float(_col(name)[i]) for name in fieldnames}
            writer.writerow(row)


if __name__ == "__main__":
    main()
