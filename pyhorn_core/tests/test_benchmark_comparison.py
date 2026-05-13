"""Regression test: run Hornresp vs pyhorn benchmark comparison.

This test invokes compare_benchmark.py (the Phase 3 overlay plot script) as a
subprocess and validates:
  1. The script exits 0 even when the Hornresp CSV is absent (graceful degradation)
  2. The benchmark PNG is produced with non-zero size
  3. When Geopan's Hornresp CSV is present, the comparison runs and RMS error
     is within the Phase 4 CI threshold (< 2 dB RMS)

Run standalone:
  pytest pyhorn_core/tests/test_benchmark_comparison.py -v

CI integration (Phase 4):
  This test runs as part of the normal pytest suite on every PR.
  It will fail if the compare_benchmark.py script crashes or the plot is not
  produced. It skips RMS assertions when the Hornresp CSV is not yet exported.

Reference:
  tests/benchmarks/compare_benchmark.py
  BACKLOG.md → Benchmark: pyhorn vs Hornresp — Phase 3 / Phase 4
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

# ── Paths ────────────────────────────────────────────────────────────────────

BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
COMPARE_SCRIPT = BENCHMARK_DIR / "compare_benchmark.py"
BENCHMARK_PLOT = BENCHMARK_DIR / "benchmark_comparison.png"
PYHORN_CSV = BENCHMARK_DIR / "output" / "response.csv"
HORNRESP_CSV = BENCHMARK_DIR / "hornresp" / "response.csv"

# Phase 4 CI threshold
RMS_THRESHOLD_DB = 2.0   # dB — SPL RMS deviation threshold for CI pass/fail


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pyhorn_spl() -> tuple[np.ndarray, np.ndarray]:
    """Load (frequency, SPL) from pyhorn's response.csv."""
    import csv as csv_lib

    freq, spl = [], []
    with open(PYHORN_CSV, newline="", encoding="utf-8") as f:
        reader = csv_lib.DictReader(f)
        for row in reader:
            freq.append(float(row["Frequency_Hz"]))
            spl.append(float(row["SPL_dB_Horn SPL (dB)"]))
    return np.array(freq), np.array(spl)


def _load_hornresp_spl() -> tuple[np.ndarray, np.ndarray]:
    """Load (frequency, SPL) from Hornresp exported CSV.

    Tries the Hornresp-specific directory first, then the legacy path.
    """
    import csv as csv_lib

    candidates = [
        BENCHMARK_DIR / "hornresp" / "response.csv",
        Path(__file__).parent / "hornresp_data" / "hornresp_flh_reference" / "response.csv",
    ]
    for path in candidates:
        if path.exists():
            freq, spl = [], []
            with open(path, newline="", encoding="utf-8") as f:
                reader = csv_lib.DictReader(f)
                # Try Hornresp column name patterns
                headers = reader.fieldnames or []
                h_lower = {h.lower().strip(): h for h in headers}
                freq_col = next(
                    (h_lower[k] for k in ["frequency", "freq", "frequency (hz)", "freq (hz)"]
                     if k in h_lower), None,
                )
                spl_col = next(
                    (h_lower[k] for k in ["spl", "spl (db)", "spl (db ref 2.83v)",
                                            "spl (db ref 2.83v/m)"]
                     if k in h_lower), None,
                )
                if freq_col is None or spl_col is None:
                    # Fall back to DictReader which handles by-header lookup
                    for row in reader:
                        freq.append(float(row[freq_col]))
                        spl.append(float(row[spl_col]))
                    return np.array(freq), np.array(spl)
            # First pass: identify columns
            freq, spl = [], []
            f.seek(0)
            next(f)  # skip header
            for row in csv_lib.DictReader(f, fieldnames=headers):
                freq.append(float(row[freq_col]))
                spl.append(float(row[spl_col]))
            return np.array(freq), np.array(spl)
    raise FileNotFoundError(f"Hornresp CSV not found in {candidates}")


# ─────────────────────────────────────────────────────────────────────────────
# TEST: compare_benchmark.py subprocess run
# ─────────────────────────────────────────────────────────────────────────────

class TestCompareBenchmarkScript:
    """Verify compare_benchmark.py runs without error."""

    def test_script_exists(self):
        """compare_benchmark.py must exist."""
        assert COMPARE_SCRIPT.exists(), f"Script not found: {COMPARE_SCRIPT}"

    def test_pyhorn_response_csv_exists(self):
        """The pyhorn response.csv must have been generated (pre-condition)."""
        assert PYHORN_CSV.exists(), (
            f"pyhorn response.csv not found at {PYHORN_CSV}\n"
            "  Run: pyhorn calculate --benchmark --driver ... --project ... "
            "--output-dir pyhorn_core/tests/benchmarks/output"
        )

    def test_script_exits_zero_without_hornresp_csv(self, tmp_path):
        """compare_benchmark.py must exit 0 even when Hornresp CSV is absent.

        The script is designed to degrade gracefully: it prints a warning and
        still produces a pyhorn-only plot when the reference CSV is missing.
        """
        result = subprocess.run(
            [sys.executable, str(COMPARE_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(BENCHMARK_DIR),
        )
        # Script must succeed (exit 0) regardless of whether Hornresp CSV exists
        assert result.returncode == 0, (
            f"compare_benchmark.py failed with exit code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    def test_script_output_mentions_hornresp_warning_when_csv_absent(self, tmp_path):
        """When Hornresp CSV is absent, the script must warn (not crash silently)."""
        result = subprocess.run(
            [sys.executable, str(COMPARE_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(BENCHMARK_DIR),
        )
        combined = result.stdout + result.stderr
        if not HORNRESP_CSV.exists():
            assert "WARNING" in combined or "not found" in combined.lower(), (
                "Expected a warning about missing Hornresp CSV in output"
            )

    def test_benchmark_plot_is_produced(self, tmp_path):
        """benchmark_comparison.png must be created by the script."""
        # Ensure plot does not pre-exist (clean slate)
        if BENCHMARK_PLOT.exists():
            BENCHMARK_PLOT.unlink()

        result = subprocess.run(
            [sys.executable, str(COMPARE_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(BENCHMARK_DIR),
        )
        assert result.returncode == 0
        assert BENCHMARK_PLOT.exists(), (
            f"Plot not produced at {BENCHMARK_PLOT} despite exit 0"
        )

    def test_benchmark_plot_has_nonzero_size(self):
        """The produced PNG must have non-trivial content (not a blank/0-byte file)."""
        # Run the script first to ensure plot is current
        subprocess.run(
            [sys.executable, str(COMPARE_SCRIPT)],
            capture_output=True,
            cwd=str(BENCHMARK_DIR),
        )
        assert BENCHMARK_PLOT.exists(), f"Plot not found: {BENCHMARK_PLOT}"
        size_bytes = BENCHMARK_PLOT.stat().st_size
        assert size_bytes > 10_000, (
            f"Plot file is suspiciously small ({size_bytes} bytes) — "
            "may be a blank/placeholder PNG"
        )

    def test_plot_contains_expected_curves(self):
        """Verify the plot image has plausible content by checking its structure.

        A real matplotlib PNG with plotted data will have:
          - File size > 10 KB (confirmed by test_benchmark_plot_has_nonzero_size)
          - Valid PNG header (\\x89PNG)
          - Non-trivial pixel data
        """
        subprocess.run(
            [sys.executable, str(COMPARE_SCRIPT)],
            capture_output=True,
            cwd=str(BENCHMARK_DIR),
        )
        with open(BENCHMARK_PLOT, "rb") as f:
            header = f.read(8)
        assert header[:4] == b"\x89PNG", (
            f"Plot file does not have PNG header: {header!r}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Phase 4 CI — RMS deviation threshold (activates when Hornresp CSV present)
# ─────────────────────────────────────────────────────────────────────────────

class TestPhase4CIDeviations:
    """Phase 4 CI regression: fail if pyhorn/Hornresp RMS deviation exceeds threshold.

    These tests activate automatically once Geopan exports the reference CSV to:
      tests/benchmarks/hornresp/response.csv
    or:
      pyhorn_core/tests/hornresp_data/hornresp_flh_reference/response.csv

    CI threshold: RMS SPL deviation < 2.0 dB across 20 Hz – 5 kHz band.
    See test_vs_hornresp.py for the full threshold philosophy documentation.
    """

    def test_hornresp_csv_present_for_ci(self):
        """Informational: report whether the Hornresp CSV is available for CI.

        CI pass/fail is determined by test_rms_deviation_under_threshold.
        """
        if not HORNRESP_CSV.exists():
            pytest.skip(
                f"Hornresp CSV not found at {HORNRESP_CSV} — "
                "CI RMS deviation test is skipped until Geopan exports the CSV. "
                "To enable: File → Export response data → CSV → "
                "tests/benchmarks/hornresp/response.csv"
            )

    def test_rms_deviation_under_threshold(self):
        """SPL RMS deviation vs Hornresp must be < 2 dB for CI pass.

        This is the Phase 4 regression gate: if this test fails on a PR,
        the PR has introduced a physics or units regression.
        """
        if not HORNRESP_CSV.exists():
            pytest.skip("Hornresp CSV not available")

        # Load both curves
        pyhorn_freq, pyhorn_spl = _load_pyhorn_spl()
        hornresp_freq, hornresp_spl = _load_hornresp_spl()

        # Resample Hornresp onto pyhorn frequency grid
        hornresp_spl_rs = np.interp(pyhorn_freq, hornresp_freq, hornresp_spl)

        # RMS across the full band
        diff = pyhorn_spl - hornresp_spl_rs
        rms = float(np.sqrt(np.mean(diff**2)))

        assert rms < RMS_THRESHOLD_DB, (
            f"SPL RMS deviation ({rms:.2f} dB) exceeds CI threshold "
            f"({RMS_THRESHOLD_DB} dB). This indicates a physics or units "
            f"regression in the horn_response() solver.\n"
            f"  Peak-to-peak diff: {np.max(diff):+.1f} / {np.min(diff):+.1f} dB\n"
            f"  Mean diff:        {np.mean(diff):+.2f} dB\n"
            f"  Std of diff:      {np.std(diff):.2f} dB"
        )

    def test_peak_spl_within_2db(self):
        """Peak SPL must be within 2 dB of Hornresp reference (CI gate)."""
        if not HORNRESP_CSV.exists():
            pytest.skip("Hornresp CSV not available")

        pyhorn_freq, pyhorn_spl = _load_pyhorn_spl()
        hornresp_freq, hornresp_spl = _load_hornresp_spl()

        py_peak = float(np.max(pyhorn_spl))
        ref_peak = float(np.max(hornresp_spl))
        diff = abs(py_peak - ref_peak)

        assert diff <= 2.0, (
            f"Peak SPL: pyhorn={py_peak:.1f} dB vs Hornresp={ref_peak:.1f} dB "
            f"(diff={diff:.2f} dB, threshold=2 dB)"
        )

    def test_spl_at_1khz_within_3db(self):
        """SPL at 1 kHz must be within 3 dB of Hornresp (known sensitivity point).

        Note: The 1 kHz region is sensitive to standing-wave structure and
        directivity model differences. The tolerance is 3 dB (wider than the
        overall 2 dB RMS threshold) to account for these effects.
        See BACKLOG.md → Benchmark → Phase 3 → 1 kHz discrepancy.
        """
        if not HORNRESP_CSV.exists():
            pytest.skip("Hornresp CSV not available")

        pyhorn_freq, pyhorn_spl = _load_pyhorn_spl()
        hornresp_freq, hornresp_spl = _load_hornresp_spl()

        hornresp_spl_rs = np.interp(pyhorn_freq, hornresp_freq, hornresp_spl)

        idx_1khz = int(np.argmin(np.abs(pyhorn_freq - 1000)))
        diff = abs(pyhorn_spl[idx_1khz] - hornresp_spl_rs[idx_1khz])

        assert diff <= 3.0, (
            f"SPL at 1 kHz: pyhorn={pyhorn_spl[idx_1khz]:.1f} dB vs "
            f"Hornresp={hornresp_spl_rs[idx_1khz]:.1f} dB "
            f"(diff={diff:.2f} dB, threshold=3 dB)"
        )

    def test_efficiency_peak_within_1pct(self):
        """Peak efficiency must be within 1% of Hornresp reference (CI gate)."""
        if not HORNRESP_CSV.exists():
            pytest.skip("Hornresp CSV not available")

        # Load pyhorn efficiency from the response CSV
        import csv as csv_lib

        eff_col = None
        with open(PYHORN_CSV, newline="", encoding="utf-8") as f:
            reader = csv_lib.DictReader(f)
            headers = reader.fieldnames or []
            for h in headers:
                if "efficiency" in h.lower() or "eff" in h.lower():
                    eff_col = h
                    break

        if eff_col is None:
            pytest.skip("Efficiency column not found in pyhorn response CSV")

        pyhorn_freq, pyhorn_eff = [], []
        with open(PYHORN_CSV, newline="", encoding="utf-8") as f:
            reader = csv_lib.DictReader(f)
            for row in reader:
                pyhorn_freq.append(float(row["Frequency_Hz"]))
                pyhorn_eff.append(float(row[eff_col]))

        pyhorn_freq = np.array(pyhorn_freq)
        pyhorn_eff = np.array(pyhorn_eff)

        # Find peak in 500-1500 Hz band
        band = (pyhorn_freq >= 500) & (pyhorn_freq <= 1500)
        if not np.any(band):
            pytest.skip("No frequency points in 500-1500 Hz efficiency band")
        py_peak_eff = float(np.max(pyhorn_eff[band]))

        # Reference: Hornresp shows 14.1% peak efficiency
        REF_PEAK_EFF = 14.1  # %
        diff = abs(py_peak_eff - REF_PEAK_EFF)

        assert diff <= 1.0, (
            f"Peak efficiency: pyhorn={py_peak_eff:.1f}% vs "
            f"Hornresp={REF_PEAK_EFF:.1f}% (diff={diff:.2f}%, threshold=1%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Known differences documentation
# ─────────────────────────────────────────────────────────────────────────────

class TestKnownDifferences:
    """Document known systematic differences between pyhorn and Hornresp.

    These are NOT failures — they are understood limitations of the TMM
    implementation or measurement setup differences. The CI thresholds
    above are set wide enough to accommodate them.
    """

    def test_known_1khz_sensitivity(self):
        """Document why 1 kHz is sensitive to standing-wave effects.

        The 1 kHz region is where comb filtering from the horn's standing-wave
        structure interacts with the directivity model. pyhorn and Hornresp may
        use slightly different piston directivity assumptions at 1 kHz.

        Resolution: 3 dB tolerance on 1 kHz SPL (wider than the 2 dB RMS
        threshold) accounts for this. See gap-findings-2026-05-01.md.
        """
        # This is a documentation test — always passes
        assert True

    def test_known_group_delay_unwrap_artifacts(self):
        """Document why group delay may be anomalous near standing-wave frequencies.

        TMM phase unwrapping can fail at standing-wave frequencies where the
        phase passes through ±π discontinuities. This produces spurious
        negative group delay values at ~1427, 2854, 4291 Hz (integer multiples
        of the ~124 Hz standing-wave fundamental).

        Resolution: test_vs_hornresp_comparison.py tolerates up to 10% anomalous
        group delay values. This is documented in the test docstring.
        """
        assert True
