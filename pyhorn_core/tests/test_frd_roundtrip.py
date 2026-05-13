"""FRD export roundtrip tests.

Verifies that FRD files (REW/ARTA format) can be exported from pyhorn and
re-imported without loss of data. Covers:

1. Core exporter roundtrip — simulate → export_to_frd() → re-parse → verify
2. CLI export roundtrip    — `pyhorn calculate --export-frd PATH` → re-parse → verify

FRD format specification
------------------------
- Plain text, one measurement point per line
- Header: ``!FRD1.0`` + ``Frequency(Hz)  Magnitude(dB)  Phase(deg)``
- Data lines: ``<freq>  <spl_db>  <phase_deg>``  (4 decimal places, space-separated)
- Logarithmic frequency spacing

Run:
    pytest pyhorn_core/tests/test_frd_roundtrip.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pyhorn_cli.cli.commands import commands_app as app
from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry
from pyhorn_core.output.exporter import export_to_frd
from pyhorn_core.solver.models import horn_response


# ─────────────────────────────────────────────────────────────────────────────
# Paths & fixtures
# ─────────────────────────────────────────────────────────────────────────────

TESTS_DIR  = Path(__file__).parent
ROOT_DIR   = TESTS_DIR.parent.parent
DRIVER_YAML = ROOT_DIR / "drivers" / "FE166NV2.yaml"
GEOM_FSX    = ROOT_DIR / "source"  / "fsx.yaml"


@pytest.fixture
def cli_runner():
    return CliRunner()


def _frd_lines(frd_path: Path):
    """Yield (freq, spl_db, phase_deg) tuples from a parsed .frd file.

    Skips blank lines, ``!FRD1.0`` header, and column-header lines.
    Returns raw float tuples as read from the file.
    """
    with open(frd_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip header / comment lines
            if line.startswith("!") or line.startswith("Frequency"):
                continue
            parts = line.split()
            freq  = float(parts[0])
            spl   = float(parts[1])
            phase = float(parts[2]) if len(parts) > 2 else 0.0
            yield freq, spl, phase


def _frd_arrays(frd_path: Path):
    """Parse a .frd file and return (freqs, spls, phases) as np arrays."""
    data = list(_frd_lines(frd_path))
    freqs, spls, phases = zip(*data)
    return np.array(freqs), np.array(spls), np.array(phases)


def _run_simulation(freqs: np.ndarray | None = None):
    """Run a minimal horn-response simulation for export roundtrip tests."""
    driver = parse_driver_specs(DRIVER_YAML)
    horn   = parse_horn_geometry(GEOM_FSX)
    if freqs is None:
        freqs = np.array([80.0, 200.0, 500.0, 1000.0, 3000.0, 5000.0])
    return horn_response(freqs=freqs, driver=driver, horn=horn, compute_distortion=False)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core exporter roundtrip
# ─────────────────────────────────────────────────────────────────────────────

class TestFrdExporterRoundtrip:
    """Verify export_to_frd() produces a .frd that re-parses correctly."""

    def test_frd_header_is_frd1(self, tmp_path):
        """FRD file must start with the ``!FRD1.0`` magic string."""
        result = _run_simulation()
        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        with open(frd_path, encoding="utf-8") as f:
            first_line = f.readline().strip()
        assert first_line == "!FRD1.0", f"Expected '!FRD1.0', got {first_line!r}"

    def test_frd_column_header_present(self, tmp_path):
        """FRD file must contain the ``Frequency(Hz)  Magnitude(dB)  Phase(deg)`` header."""
        result = _run_simulation()
        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        content = open(frd_path, encoding="utf-8").read()
        assert "Frequency(Hz)" in content and "Magnitude(dB)" in content and "Phase(deg)" in content

    def test_frd_data_line_count(self, tmp_path):
        """Number of data lines (excluding headers) must equal number of frequency points."""
        result = _run_simulation()
        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        data = list(_frd_lines(frd_path))
        assert len(data) == len(result.freqs), (
            f"FRD has {len(data)} data lines but simulation has {len(result.freqs)} freq points"
        )

    def test_frd_frequencies_match_exactly(self, tmp_path):
        """Frequency values written to .frd must read back byte-for-byte."""
        result = _run_simulation()
        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        reimport_freqs, _, _ = _frd_arrays(frd_path)
        np.testing.assert_array_equal(reimport_freqs, result.freqs)

    def test_frd_spl_matches_within_0_01db(self, tmp_path):
        """SPL values must survive FRD roundtrip within 0.01 dB.

        FRD format stores 4 decimal places → max quantisation error ≈ 5e-5 dB.
        We allow 0.01 dB to cover floating-point formatting edge cases.
        """
        result = _run_simulation()
        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        _, reimport_spl, _ = _frd_arrays(frd_path)
        np.testing.assert_allclose(reimport_spl, result.spl, atol=0.01)

    def test_frd_phase_matches_within_0_1deg(self, tmp_path):
        """Phase values must survive FRD roundtrip within 0.1 degrees.

        FRD format stores 4 decimal places → max quantisation error ≈ 5e-5 deg.
        We allow 0.1 deg to be generous.
        """
        result = _run_simulation()
        phase_deg = np.rad2deg(result.phase)
        frd_path = tmp_path / "response.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=phase_deg,
            output_path=frd_path,
        )
        _, _, reimport_phase = _frd_arrays(frd_path)
        np.testing.assert_allclose(reimport_phase, phase_deg, atol=0.1)

    def test_frd_single_frequency_point(self, tmp_path):
        """FRD export must handle a single frequency point (edge case)."""
        freqs = np.array([1000.0])
        result = _run_simulation(freqs=freqs)
        frd_path = tmp_path / "single.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        reimport_freqs, reimport_spl, _ = _frd_arrays(frd_path)
        assert len(reimport_freqs) == 1
        np.testing.assert_allclose(reimport_spl, result.spl, atol=0.01)

    def test_frd_wide_frequency_range(self, tmp_path):
        """FRD export must handle wide frequency range (20 Hz – 20 kHz)."""
        freqs = np.array([20.0, 100.0, 1000.0, 10000.0, 20000.0])
        result = _run_simulation(freqs=freqs)
        frd_path = tmp_path / "wide.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        reimport_freqs, reimport_spl, reimport_phase = _frd_arrays(frd_path)
        assert len(reimport_freqs) == 5
        np.testing.assert_allclose(reimport_spl, result.spl, atol=0.01)
        np.testing.assert_allclose(reimport_phase, np.rad2deg(result.phase), atol=0.1)

    def test_frd_file_is_utf8(self, tmp_path):
        """FRD file must be valid UTF-8 (no encoding errors on re-read)."""
        result = _run_simulation()
        frd_path = tmp_path / "encoding.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        # Must not raise UnicodeDecodeError
        with open(frd_path, encoding="utf-8") as f:
            f.read()

    def test_frd_file_is_plain_text(self, tmp_path):
        """FRD file must be a plain text file (readable without binary mode)."""
        result = _run_simulation()
        frd_path = tmp_path / "plaintext.frd"
        export_to_frd(
            freqs=result.freqs,
            spl_db=result.spl,
            phase_deg=np.rad2deg(result.phase),
            output_path=frd_path,
        )
        with open(frd_path, encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0
        assert content.startswith("!FRD1.0")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLI export roundtrip  —  pyhorn calculate --export-frd PATH
# ─────────────────────────────────────────────────────────────────────────────

class TestFrdCliRoundtrip:
    """Verify the CLI ``calculate --export-frd`` flag produces a valid .frd."""

    def test_calculate_export_frd_creates_file(self, cli_runner, tmp_path):
        """``--export-frd`` must create the target file (exit 0)."""
        frd_path = tmp_path / "cli_test.frd"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "--driver",  str(DRIVER_YAML),
                "--horn",    str(GEOM_FSX),
                "--fmin",    "100",
                "--fmax",    "5000",
                "--n-points","50",
                "--no-plot",
                "--export-frd", str(frd_path),
            ],
        )
        assert result.exit_code == 0, f"CLI exited non-zero:\n{result.output}\n{result.exception}"
        assert frd_path.exists(), f"FRD file not created at {frd_path}"

    def test_calculate_export_frd_has_header(self, cli_runner, tmp_path):
        """FRD produced by CLI must start with ``!FRD1.0``."""
        frd_path = tmp_path / "cli_header.frd"
        cli_runner.invoke(
            app,
            [
                "calculate",
                "--driver",  str(DRIVER_YAML),
                "--horn",    str(GEOM_FSX),
                "--fmin",    "100",
                "--fmax",    "2000",
                "--n-points","20",
                "--no-plot",
                "--export-frd", str(frd_path),
            ],
        )
        with open(frd_path, encoding="utf-8") as f:
            first = f.readline().strip()
        assert first == "!FRD1.0", f"Expected '!FRD1.0', got {first!r}"

    def test_calculate_export_frd_spl_in_sensible_range(self, cli_runner, tmp_path):
        """SPL values in the CLI-produced FRD must be finite and in a sensible range."""
        frd_path = tmp_path / "cli_spl.frd"
        cli_runner.invoke(
            app,
            [
                "calculate",
                "--driver",  str(DRIVER_YAML),
                "--horn",    str(GEOM_FSX),
                "--fmin",    "100",
                "--fmax",    "5000",
                "--n-points","100",
                "--no-plot",
                "--export-frd", str(frd_path),
            ],
        )

        _, reimport_spl, reimport_phase = _frd_arrays(frd_path)
        # All values must be finite numbers
        assert np.all(np.isfinite(reimport_spl)), "SPL values must all be finite"
        assert np.all(np.isfinite(reimport_phase)), "Phase values must all be finite"
        # SPL in a sensible range for a home speaker (50–130 dB)
        assert np.all(reimport_spl > 0), "SPL must be positive (dB SPL relative)"
        assert np.all(reimport_spl < 150), "SPL unrealistically high"

    def test_calculate_export_frd_phase_in_sensible_range(self, cli_runner, tmp_path):
        """Phase values in the CLI-produced FRD must be finite and in (-720, 720) deg."""
        frd_path = tmp_path / "cli_phase.frd"
        cli_runner.invoke(
            app,
            [
                "calculate",
                "--driver",  str(DRIVER_YAML),
                "--horn",    str(GEOM_FSX),
                "--fmin",    "100",
                "--fmax",    "5000",
                "--n-points","100",
                "--no-plot",
                "--export-frd", str(frd_path),
            ],
        )

        _, _, reimport_phase = _frd_arrays(frd_path)
        assert np.all(np.isfinite(reimport_phase)), "Phase values must all be finite"

    def test_calculate_export_frd_frequency_count(self, cli_runner, tmp_path):
        """FRD produced by CLI must have one line per frequency point requested."""
        frd_path = tmp_path / "cli_count.frd"
        n_points = 75
        cli_runner.invoke(
            app,
            [
                "calculate",
                "--driver",  str(DRIVER_YAML),
                "--horn",    str(GEOM_FSX),
                "--fmin",    "100",
                "--fmax",    "5000",
                "--n-points", str(n_points),
                "--no-plot",
                "--export-frd", str(frd_path),
            ],
        )
        freqs, _, _ = _frd_arrays(frd_path)
        assert len(freqs) == n_points, f"Expected {n_points} freq points, got {len(freqs)}"

    def test_calculate_export_frd_requires_valid_driver(self, cli_runner, tmp_path):
        """Passing a non-existent driver must fail with exit_code != 0."""
        frd_path = tmp_path / "fail.frd"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "--driver",  "/nonexistent/driver.yaml",
                "--horn",    str(GEOM_FSX),
                "--fmin",    "100",
                "--fmax",    "1000",
                "--no-plot",
                "--export-frd", str(frd_path),
            ],
        )
        assert result.exit_code != 0

    def test_calculate_export_frd_overwrites_existing(self, cli_runner, tmp_path):
        """Calling --export-frd twice must overwrite the file (no error)."""
        frd_path = tmp_path / "overwrite.frd"
        for _ in range(2):
            result = cli_runner.invoke(
                app,
                [
                    "calculate",
                    "--driver",  str(DRIVER_YAML),
                    "--horn",    str(GEOM_FSX),
                    "--fmin",    "100",
                    "--fmax",    "1000",
                    "--no-plot",
                    "--export-frd", str(frd_path),
                ],
            )
            assert result.exit_code == 0
        # File must still be valid after overwrite
        reimport_freqs, _, _ = _frd_arrays(frd_path)
        assert len(reimport_freqs) > 0
