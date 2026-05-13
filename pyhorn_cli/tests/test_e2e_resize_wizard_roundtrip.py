"""E2E roundtrip test: resize-wizard output → calculate.

Verifies that the geometry YAML produced by resize-wizard is valid input for
the acoustic simulation (calculate), producing finite SPL values in a plausible
dB range.
"""

from __future__ import annotations

import csv
import math
import subprocess
from pathlib import Path

import pytest

# The pyhorn CLI is installed as a standalone entry-point script (not as a
# module). Use it directly so we pick up the correct Python interpreter.
PYHORN_CLI = "/opt/homebrew/bin/pyhorn"

WORKDIR = Path("/Users/guillaume/pyhorn")
DRIVER_YAML = WORKDIR / "pyhorn_core/tests/benchmarks/hornresp_reference_driver.yaml"
HORN_YAML = WORKDIR / "pyhorn_core/tests/benchmarks/hornresp_reference_flh.yaml"


class TestResizeWizardRoundtrip:
    """resize-wizard output must be simulatable via `calculate`."""

    def test_resize_wizard_output_feeds_calculate(
        self, tmp_path: Path
    ) -> None:
        """
        Run resize-wizard then calculate; verify:
          1. Both commands exit 0
          2. response.csv is created
          3. SPL values are finite and in 60–130 dB range
          4. No NaN or Inf values in the SPL column
        """
        resized_yaml = tmp_path / "resized_geometry.yaml"
        calc_out_dir = tmp_path / "calc_out"

        # Step 1: resize-wizard
        result_resize = subprocess.run(
            [
                PYHORN_CLI, "resize-wizard",
                "--driver", str(DRIVER_YAML),
                "--horn", str(HORN_YAML),
                "--factor", "1.2",
                "--geometry-only",
                "--output", str(resized_yaml),
            ],
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result_resize.returncode == 0, (
            f"resize-wizard failed (exit {result_resize.returncode}):\n"
            f"STDOUT:\n{result_resize.stdout}\n"
            f"STDERR:\n{result_resize.stderr}"
        )
        assert resized_yaml.exists(), (
            f"resize-wizard did not produce output file: {resized_yaml}\n"
            f"stdout: {result_resize.stdout}"
        )

        # Step 2: calculate
        result_calc = subprocess.run(
            [
                PYHORN_CLI, "calculate",
                "--driver", str(DRIVER_YAML),
                "--horn", str(resized_yaml),
                "--output-dir", str(calc_out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "20",
                "--no-plot",
            ],
            cwd=str(WORKDIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result_calc.returncode == 0, (
            f"calculate failed (exit {result_calc.returncode}):\n"
            f"STDOUT:\n{result_calc.stdout}\n"
            f"STDERR:\n{result_calc.stderr}"
        )

        # Step 3: locate response.csv
        csv_files = list(calc_out_dir.rglob("response.csv"))
        assert len(csv_files) == 1, (
            f"Expected 1 response.csv in {calc_out_dir}, found: {csv_files}\n"
            f"Contents: {list(calc_out_dir.rglob('*'))}"
        )
        response_csv = csv_files[0]

        # Step 4: parse CSV and validate SPL column
        with open(response_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 20, (
            f"Expected 20 frequency points, got {len(rows)}"
        )

        # Find the Horn SPL column
        spl_col = next(
            (col for col in rows[0].keys() if "SPL" in col and "Horn SPL" in col),
            None,
        )
        assert spl_col is not None, (
            f"No 'Horn SPL' column found in CSV header: {list(rows[0].keys())}"
        )

        spl_values = [float(r[spl_col]) for r in rows]

        # Step 5: no NaN / Inf
        for i, spl in enumerate(spl_values):
            assert math.isfinite(spl), (
                f"Non-finite SPL at row {i}: {spl} "
                f"(frequency={rows[i].get('Frequency_Hz')})"
            )

        # Step 6: reasonable dB range (60–130 dB for a horn-loaded small driver)
        for i, spl in enumerate(spl_values):
            assert 60.0 <= spl <= 130.0, (
                f"SPL value {spl:.2f} dB at "
                f"freq={rows[i].get('Frequency_Hz')} Hz is outside "
                f"plausible range [60, 130] dB"
            )
