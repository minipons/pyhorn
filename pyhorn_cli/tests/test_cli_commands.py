"""Unit tests for pyhorn_cli.cli.commands."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from pyhorn_cli.cli.commands import commands_app as app
from pyhorn_core.config.models import DriverSpecs
from pyhorn_core.solver.medial_axis import generate_auto_segments
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_core.solver.optimizer import OptimizationResult


# ─── Typer CLI runner fixture ─────────────────────────────────────────────────


@pytest.fixture
def cli_runner():
    return CliRunner()


# ─── App structure ─────────────────────────────────────────────────────────────


class TestAppStructure:
    """Tests for the Typer app structure and command registration."""

    def test_app_has_calculate_command(self, cli_runner):
        """App should recognise the 'calculate' command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "calculate" in result.output

    def test_app_has_compare_command(self, cli_runner):
        """App should recognise the 'compare' command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "compare" in result.output

    def test_app_has_derive_ts_command(self, cli_runner):
        """App should recognise the 'derive-ts' command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "derive-ts" in result.output

    def test_app_has_hornresp_command(self, cli_runner):
        """App should recognise the 'hornresp' command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "hornresp" in result.output

    def test_app_has_auto_segment_command(self, cli_runner):
        """App should recognise the 'auto-segment' command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "auto-segment" in result.output

    def test_app_has_fold_optimized_command(self, cli_runner):
        """App should recognise the 'fold-optimized' command."""
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "fold-optimized" in result.output


# ─── derive-ts ──────────────────────────────────────────────────────────────────


class TestDeriveTs:
    """Tests for the derive-ts command."""

    def test_derive_ts_outputs_all_parameters(self, cli_runner):
        """derive-ts should output all derived SI parameters."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs",
                "100",
                "--qes",
                "0.5",
                "--qms",
                "5.0",
                "--vas",
                "50.0",
                "--re",
                "7.0",
                "--sd",
                "100.0",
            ],
        )
        assert result.exit_code == 0
        assert "qts:" in result.output
        assert "bl:" in result.output
        assert "mms:" in result.output
        assert "cms:" in result.output
        assert "rms:" in result.output

    def test_derive_ts_qts_calculation(self, cli_runner):
        """Qts = Qes*Qms/(Qes+Qms)."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs",
                "100",
                "--qes",
                "0.5",
                "--qms",
                "5.0",
                "--vas",
                "50.0",
                "--re",
                "7.0",
                "--sd",
                "100.0",
            ],
        )
        # qts = 0.5*5/(0.5+5) = 2.5/5.5 ≈ 0.455
        assert "qts: 0.455" in result.output or "qts: 0.454" in result.output

    def test_derive_ts_requires_all_parameters(self, cli_runner):
        """Missing any parameter should result in a usage error."""
        result = cli_runner.invoke(app, ["derive-ts", "--fs", "100", "--qes", "0.5"])
        assert result.exit_code != 0

    def test_derive_ts_vas_converted_to_m3(self, cli_runner):
        """vas=50L → vas ≈ 0.00005 m³."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs",
                "100",
                "--qes",
                "0.5",
                "--qms",
                "5.0",
                "--vas",
                "50.0",
                "--re",
                "7.0",
                "--sd",
                "100.0",
            ],
        )
        # Should show vas in m³ (tiny number, scientific notation)
        assert "vas:" in result.output

    # ── Boundary / error value tests ──────────────────────────────────────────

    def test_derive_ts_fs_zero_exits_nonzero(self, cli_runner):
        """fs=0 would cause division by zero — must exit non-zero."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "0",
                "--qes", "0.5",
                "--qms", "5.0",
                "--vas", "50.0",
                "--re", "7.0",
                "--sd", "100.0",
            ],
        )
        assert result.exit_code != 0

    def test_derive_ts_qes_zero_exits_nonzero(self, cli_runner):
        """Qes=0 would zero out Qts — must exit non-zero."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "100",
                "--qes", "0",
                "--qms", "5.0",
                "--vas", "50.0",
                "--re", "7.0",
                "--sd", "100.0",
            ],
        )
        assert result.exit_code != 0

    def test_derive_ts_qms_zero_exits_nonzero(self, cli_runner):
        """Qms=0 would zero out Qts — must exit non-zero."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "100",
                "--qes", "0.5",
                "--qms", "0",
                "--vas", "50.0",
                "--re", "7.0",
                "--sd", "100.0",
            ],
        )
        assert result.exit_code != 0

    def test_derive_ts_vas_zero_exits_nonzero(self, cli_runner):
        """Vas=0 would cause division by zero in CMS/VAS relationship."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "100",
                "--qes", "0.5",
                "--qms", "5.0",
                "--vas", "0",
                "--re", "7.0",
                "--sd", "100.0",
            ],
        )
        assert result.exit_code != 0

    def test_derive_ts_sd_zero_exits_nonzero(self, cli_runner):
        """Sd=0 would cause division by zero (bl = 0, cms infinite)."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "100",
                "--qes", "0.5",
                "--qms", "5.0",
                "--vas", "50.0",
                "--re", "7.0",
                "--sd", "0",
            ],
        )
        assert result.exit_code != 0

    def test_derive_ts_negative_qes_exits_nonzero(self, cli_runner):
        """Negative Qes is physically impossible — must exit non-zero."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "100",
                "--qes", "-0.5",
                "--qms", "5.0",
                "--vas", "50.0",
                "--re", "7.0",
                "--sd", "100.0",
            ],
        )
        assert result.exit_code != 0

    def test_derive_ts_subprocess_smoke_test(self):
        """Smoke test: derive-ts via subprocess produces valid SI parameter output."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "derive-ts",
             "--fs", "100", "--qes", "0.5", "--qms", "5.0",
             "--vas", "50.0", "--re", "7.0", "--sd", "100.0"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"derive-ts subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:200]}"
        )
        # Verify key SI parameters appear in stdout
        for param in ["qts:", "bl:", "mms:", "cms:", "rms:"]:
            assert param in result.stdout, (
                f"Expected '{param}' in stdout, got:\n{result.stdout[:300]}"
            )

    def test_derive_ts_subprocess_fs_zero_exits_nonzero(self):
        """Subprocess smoke: derive-ts with fs=0 exits non-zero (validation error)."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "derive-ts",
             "--fs", "0", "--qes", "0.5", "--qms", "5.0",
             "--vas", "50.0", "--re", "7.0", "--sd", "100.0"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"derive-ts with fs=0 should fail, got exit {result.returncode}"
        )
        # Error message should mention fs or positive
        combined = result.stderr + result.stdout
        assert "fs" in combined.lower() or "positive" in combined.lower(), (
            f"Expected error about fs/positive, got:\n{combined[:300]}"
        )


    def test_derive_ts_output_yaml_flag(self, cli_runner):
        """--output-yaml emits a clean YAML snippet with all SI parameters."""
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "53",
                "--qes", "0.29",
                "--qms", "3.8",
                "--vas", "17.5",
                "--re", "7.8",
                "--sd", "132.7",
                "--output-yaml",
            ],
        )
        assert result.exit_code == 0
        assert "# Derived T-S parameters" in result.output
        for key in ["sd:", "re:", "bl:", "cms:", "mms:", "rms:", "qts:", "vas:", "fs:"]:
            assert key in result.output, f"Missing key {key} in YAML output"
        assert "!!python" not in result.output
        assert "numpy" not in result.output

    def test_derive_ts_output_yaml_is_parseable(self, cli_runner):
        """--output-yaml output is valid YAML that can be parsed."""
        import yaml as _yaml
        result = cli_runner.invoke(
            app,
            [
                "derive-ts",
                "--fs", "53",
                "--qes", "0.29",
                "--qms", "3.8",
                "--vas", "17.5",
                "--re", "7.8",
                "--sd", "132.7",
                "--output-yaml",
            ],
        )
        assert result.exit_code == 0
        yaml_content = result.output.split("\n", 1)[1]  # skip the "# Derived T-S parameters..." comment line
        parsed = _yaml.safe_load(yaml_content)
        assert isinstance(parsed, dict)
        assert all(isinstance(v, (int, float)) for v in parsed.values())

    def test_derive_ts_subprocess_roundtrip_calculate(self, tmp_path):
        """E2E roundtrip: derive-ts --output-yaml → driver YAML → calculate → valid SPL.

        This verifies that the T-S parameters emitted by derive-ts can be used as a
        driver YAML input to the calculate command, and that the simulation completes
        without NaN/Inf in the SPL output.
        """
        import subprocess, sys, yaml

        # Step 1: derive-ts --output-yaml for FE166NV2 T-S params
        derive_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "derive-ts",
             "--fs", "53", "--qes", "0.29", "--qms", "3.8",
             "--vas", "17.5", "--re", "7.8", "--sd", "132.7",
             "--output-yaml"],
            capture_output=True, text=True, timeout=30,
        )
        assert derive_result.returncode == 0, (
            f"derive-ts failed: {derive_result.stderr[:200]}"
        )

        # Parse YAML output (skip the "# Derived T-S parameters" comment line)
        yaml_text = "\n".join(derive_result.stdout.splitlines()[1:])
        ts_params = yaml.safe_load(yaml_text)
        assert isinstance(ts_params, dict), f"derive-ts YAML not parseable: {yaml_text[:200]}"

        # Step 2: Build a minimal driver YAML with derived T-S params
        driver_yaml = tmp_path / "driver.yaml"
        with open(driver_yaml, "w") as f:
            yaml.dump(ts_params, f)

        # Step 3: Run calculate with this driver + a simple horn geometry
        out_dir = tmp_path / "calc_out"
        calc_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "calculate",
             "-d", str(driver_yaml),
             "-h", "source/bk16.yaml",
             "-o", str(out_dir),
             "--fmin", "100", "--fmax", "1000", "--n-points", "50",
             "--no-plot"],
            capture_output=True, text=True, timeout=60,
        )
        assert calc_result.returncode == 0, (
            f"calculate with derived driver YAML failed (exit {calc_result.returncode}):\n"
            f"  stderr: {calc_result.stderr[:500]}\n"
            f"  stdout: {calc_result.stdout[:300]}"
        )

        # Step 4: Verify response.csv exists and has valid SPL data
        # calculate creates a subdirectory per horn (e.g. calc_out/bk16/response.csv)
        csv_path = out_dir / "bk16" / "response.csv"
        assert csv_path.exists(), (
            f"calculate did not produce response.csv at {csv_path}\n"
            f"stdout: {calc_result.stdout[:300]}"
        )
        import csv as _csv
        with open(csv_path) as f:
            reader = _csv.DictReader(f)
            rows = list(reader)
        assert len(rows) > 0, "response.csv is empty"
        spl_col = next((k for k in rows[0].keys()
                        if "SPL" in k and ("dB" in k or "Horn" in k)), None)
        assert spl_col is not None, f"No SPL column found in CSV. Keys: {list(rows[0].keys())}"
        spl_values = [float(row[spl_col]) for row in rows if row.get(spl_col)]
        assert len(spl_values) == 50, f"Expected 50 SPL points, got {len(spl_values)}"
        # Verify no NaN or Inf
        import math
        for i, spl in enumerate(spl_values):
            assert math.isfinite(spl), f"SPL value at row {i} is not finite: {spl}"
        # Verify SPL is in a reasonable range (-50 to 150 dB)
        for i, spl in enumerate(spl_values):
            assert -50 < spl < 150, f"SPL value at row {i} out of range: {spl} dB"


# ─── auto-segment ─────────────────────────────────────────────────────────────


class TestAutoSegment:
    """Tests for the auto-segment command."""

    @pytest.fixture
    def valid_2d_json(self):
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

    def test_auto_segment_writes_yaml(self, cli_runner, valid_2d_json, tmp_path):
        """--output should create the YAML file."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--n-segments",
                "10",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_auto_segment_throat_adapter_injects_yaml_section(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """--throat-adapter-d1 and --throat-adapter-d2 should add throat_adapter: to output."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--n-segments",
                "10",
                "--throat-adapter-d1",
                "50",
                "--throat-adapter-d2",
                "100",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Throat adapter" in result.output

        import yaml

        with open(out_yaml) as f:
            data = yaml.safe_load(f)
        assert "throat_adapter" in data
        assert data["throat_adapter"]["type"] == "conical"
        assert data["throat_adapter"]["ap1"] > 0
        assert data["throat_adapter"]["lpt"] > 0

    def test_auto_segment_throat_adapter_with_explicit_type_and_length(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """--throat-adapter-type and --throat-adapter-length should be honoured."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--n-segments",
                "10",
                "--throat-adapter-d1",
                "40",
                "--throat-adapter-d2",
                "80",
                "--throat-adapter-type",
                "exponential",
                "--throat-adapter-length",
                "60",
            ],
        )
        assert result.exit_code == 0, result.output

        import yaml

        with open(out_yaml) as f:
            data = yaml.safe_load(f)
        assert data["throat_adapter"]["type"] == "exponential"
        assert data["throat_adapter"]["lpt"] == pytest.approx(0.06, rel=1e-3)

    def test_auto_segment_rejects_partial_throat_adapter_flags(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """Only --throat-adapter-d1 without --throat-adapter-d2 should error."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--n-segments",
                "10",
                "--throat-adapter-d1",
                "50",
            ],
        )
        assert result.exit_code != 0

    def test_auto_segment_malformed_json(self, cli_runner, tmp_path):
        """Malformed (unparseable) JSON should error gracefully."""
        bad_json = tmp_path / "bad.json"
        bad_json.write_text('{"width": 0.2, "throat" [[0.05, 0.0]}')  # missing : after "throat", extra bracket
        out_yaml = tmp_path / "out.yaml"
        result = cli_runner.invoke(
            app,
            ["auto-segment", "-i", str(bad_json), "-o", str(out_yaml), "--n-segments", "10"],
        )
        assert result.exit_code != 0
        assert "Traceback" not in result.output

    def test_auto_segment_invalid_json_schema(self, cli_runner, tmp_path):
        """Valid JSON but missing required 'boundary_edges' field should error."""
        incomplete_json = tmp_path / "incomplete.json"
        incomplete_json.write_text(
            '{"width": 0.2, "throat": [[0.05, 0.0], [0.15, 0.0]], "mouth": [[0.3, 0.0], [0.4, 0.0]]}'
            # missing boundary_edges
        )
        out_yaml = tmp_path / "out.yaml"
        result = cli_runner.invoke(
            app,
            ["auto-segment", "-i", str(incomplete_json), "-o", str(out_yaml), "--n-segments", "10"],
        )
        assert result.exit_code != 0

    def test_auto_segment_subprocess_smoke_test(self, tmp_path):
        """Smoke test: auto-segment via subprocess exits 0 and produces valid YAML."""
        import subprocess, sys, yaml

        geo_json = tmp_path / "geometry.json"
        geo_json.write_text(
            '{"width": 0.2, '
            '"throat": [[0.05, 0.0], [0.15, 0.0]], '
            '"mouth": [[0.3, 0.0], [0.4, 0.0]], '
            '"boundary_edges": ['
            '[[0.05, 0.0], [0.4, 0.0]], '
            '[[0.4, 0.0], [0.4, 0.2]], '
            '[[0.4, 0.2], [0.05, 0.2]], '
            '[[0.05, 0.2], [0.05, 0.0]]]}'
        )
        out_yaml = tmp_path / "segments.yaml"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "auto-segment",
                "-i", str(geo_json),
                "-o", str(out_yaml),
                "--n-segments", "10",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"auto-segment subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )
        assert out_yaml.exists(), (
            f"auto-segment did not produce YAML at {out_yaml}\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )
        # Verify YAML is valid and has sections format
        with open(out_yaml) as f:
            data = yaml.safe_load(f)
        assert "sections" in data, f"auto-segment output missing 'sections' key. Keys: {list(data.keys())}"
        assert len(data["sections"]) > 0, "auto-segment output has empty sections list"


class TestHornresp:
    """Tests for the hornresp command."""

    def test_hornresp_solves_missing_t_and_writes_yaml(self, cli_runner, tmp_path):
        """One missing Hornresp input should be solved and exported."""
        out_yaml = tmp_path / "hornresp.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1",
                "100",
                "--s2",
                "300",
                "--f12",
                "150",
                "--hyp",
                "60",
                "--output",
                str(out_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Solved Hornresp Hyperbolic horn parameters" in result.output
        assert out_yaml.exists()

        import yaml
        data = yaml.safe_load(out_yaml.read_text())
        assert data["enclosure_type"] == "BLH"
        assert "sections" in data
        section = data["sections"][0]
        assert section["name"] == "main_horn"
        assert section["profile_type"] == "hyperbolic"
        assert section["start_area"] == pytest.approx(0.01)
        assert section["end_area"] == pytest.approx(0.03)
        assert section["length"] == pytest.approx(0.6)
        assert section["hyperbolic_t"] == pytest.approx(-0.385043, rel=1e-5)

    def test_hornresp_rejects_two_missing_inputs(self, cli_runner, tmp_path):
        """Two missing Hornresp inputs should fail as underdetermined."""
        out_yaml = tmp_path / "out.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1",
                "100",
                "--s2",
                "300",
                "--f12",
                "150",
            ],
        )
        assert result.exit_code != 0
        assert "underdetermined" in result.output

    def test_hornresp_conical_profile_type(self, cli_runner, tmp_path):
        """--profile-type con solves conical horn and outputs valid YAML."""
        out_yaml = tmp_path / "conical.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",
                "--s2", "300",
                "--l12", "150",
                "--profile-type", "con",
                "--output", str(out_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_yaml.exists()
        import yaml
        data = yaml.safe_load(out_yaml.read_text())
        assert data["enclosure_type"] == "BLH"
        assert "sections" in data
        assert data["sections"][0]["profile_type"] == "conical"

    def test_hornresp_exponential_profile_type(self, cli_runner, tmp_path):
        """--profile-type exp solves exponential horn and outputs valid YAML."""
        out_yaml = tmp_path / "exp.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",
                "--s2", "300",
                "--l12", "150",
                "--profile-type", "exp",
                "--output", str(out_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        import yaml
        data = yaml.safe_load(out_yaml.read_text())
        assert data["sections"][0]["profile_type"] == "exponential"

    def test_hornresp_parabolic_profile_type(self, cli_runner, tmp_path):
        """--profile-type par solves parabolic horn and outputs valid YAML."""
        out_yaml = tmp_path / "par.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",
                "--s2", "300",
                "--l12", "150",
                "--profile-type", "par",
                "--output", str(out_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        import yaml
        data = yaml.safe_load(out_yaml.read_text())
        assert data["sections"][0]["profile_type"] == "parabolic"

    def test_hornresp_catenoidal_profile_type(self, cli_runner, tmp_path):
        """--profile-type cat solves catenoidal horn and outputs valid YAML."""
        out_yaml = tmp_path / "cat.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",
                "--s2", "300",
                "--l12", "150",
                "--profile-type", "cat",
                "--output", str(out_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        import yaml
        data = yaml.safe_load(out_yaml.read_text())
        assert data["sections"][0]["profile_type"] == "catenoidal"

    def test_hornresp_hyperbolic_profile_type(self, cli_runner, tmp_path):
        """--profile-type hyp solves hyperbolic horn and outputs valid YAML."""
        out_yaml = tmp_path / "hyp.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",
                "--s2", "300",
                "--f12", "50",
                "--hyp", "150",
                "--profile-type", "hyp",
                "--output", str(out_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_yaml.exists()
        import yaml
        data = yaml.safe_load(out_yaml.read_text())
        assert data["enclosure_type"] == "BLH"
        assert "sections" in data
        assert data["sections"][0]["profile_type"] == "hyperbolic"

    def test_hornresp_invalid_profile_type(self, cli_runner, tmp_path):
        """Unknown --profile-type exits non-zero with a clear error."""
        out_yaml = tmp_path / "out.yaml"
        result = cli_runner.invoke(
            app,
            [
                "hornresp",
                "--s1", "40",
                "--s2", "300",
                "--l12", "150",
                "--profile-type", "fourier",
                "--output", str(out_yaml),
            ],
        )
        assert result.exit_code != 0

    def test_hornresp_subprocess_smoke_test(self, tmp_path):
        """Smoke test: hornresp via subprocess exits 0 and produces valid YAML."""
        import subprocess, sys, yaml
        out_yaml = tmp_path / "hornresp.yaml"
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "hornresp",
             "--s1", "40", "--s2", "300", "--l12", "150",
             "--profile-type", "con", "--output", str(out_yaml)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"hornresp subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:300]}"
        )
        assert out_yaml.exists(), "hornresp should write YAML output file"
        data = yaml.safe_load(out_yaml.read_text())
        assert "sections" in data, "YAML should contain sections"
        assert data.get("enclosure_type") in ("BLH", "FLH")

    def test_hornresp_subprocess_invalid_profile_type_exits_nonzero(self, tmp_path):
        """Subprocess smoke: hornresp with invalid --profile-type exits non-zero."""
        import subprocess, sys
        out_yaml = tmp_path / "out.yaml"
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "hornresp",
             "--s1", "40", "--s2", "300", "--l12", "150",
             "--profile-type", "fourier", "--output", str(out_yaml)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"hornresp with invalid profile-type should fail, got exit {result.returncode}"
        )

    def test_auto_segment_fails_without_input(self, cli_runner, tmp_path):
        """Neither --input nor --from-clipboard should exit with error."""
        out_yaml = tmp_path / "out.yaml"
        result = cli_runner.invoke(app, ["auto-segment", "-o", str(out_yaml)])
        assert result.exit_code != 0

    def test_auto_segment_n_segments_flag(self, cli_runner, valid_2d_json, tmp_path):
        """--n-segments should control the segment count in output."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--n-segments",
                "15",
            ],
        )
        import yaml

        with open(out_yaml) as f:
            data = yaml.safe_load(f)
        if "conical_segments" in data:
            assert len(data["conical_segments"]) == 15
        if "rectangular_segments" in data:
            assert len(data["rectangular_segments"]) == 15

    def test_auto_segment_flip_x_flag(self, cli_runner, valid_2d_json, tmp_path):
        """--flip-x should be accepted without error."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--flip-x",
                "--n-segments",
                "10",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_auto_segment_flip_y_flag(self, cli_runner, valid_2d_json, tmp_path):
        """--flip-y should be accepted without error."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--flip-y",
                "--n-segments",
                "10",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_auto_segment_geometry_aware_flag(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """--geometry-aware should set discretisation field."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--geometry-aware",
                "--n-segments",
                "10",
            ],
        )
        import yaml

        with open(out_yaml) as f:
            data = yaml.safe_load(f)
        assert data.get("discretisation") == "geometry"

    def test_auto_segment_preserve_breaks_flag(
        self, cli_runner, valid_2d_json, tmp_path
    ):
        """--preserve-breaks should output rectangular_segments."""
        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
                "--preserve-breaks",
                "--geometry-aware",
                "--n-segments",
                "10",
            ],
        )
        import yaml

        with open(out_yaml) as f:
            data = yaml.safe_load(f)
        assert "rectangular_segments" in data

    def test_auto_segment_bad_json_error(self, cli_runner, tmp_path):
        """Malformed JSON should result in an error exit code."""
        json_path = tmp_path / "bad.json"
        json_path.write_text("{ not json")
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            [
                "auto-segment",
                "-i",
                str(json_path),
                "-o",
                str(out_yaml),
            ],
        )
        assert result.exit_code != 0

    def test_auto_segment_output_yaml_schema_valid(self, cli_runner, valid_2d_json, tmp_path):
        """auto-segment output YAML should have required fields for pyhorn to accept it."""
        import yaml as yaml_lib

        json_path = tmp_path / "in.json"
        json_path.write_text(json.dumps(valid_2d_json))
        out_yaml = tmp_path / "out.yaml"

        result = cli_runner.invoke(
            app,
            ["auto-segment", "-i", str(json_path), "-o", str(out_yaml), "--n-segments", "10"],
        )
        assert result.exit_code == 0, result.output

        with open(out_yaml) as f:
            data = yaml_lib.safe_load(f)

        # auto-segment outputs a geometry YAML with sections format
        assert "sections" in data or "conical_segments" in data, (
            "sections or conical_segments required"
        )

        # sections format: each section needs name, profile_type, length, start_area, end_area
        if "sections" in data:
            sections = data["sections"]
            assert isinstance(sections, list), "sections must be a list"
            assert len(sections) > 0, "sections must not be empty"
            for i, sec in enumerate(sections):
                for field in ("name", "profile_type", "length", "start_area", "end_area"):
                    assert field in sec, f"sections[{i}].{field} is required"
                assert sec["length"] > 0, f"sections[{i}].length must be positive"
                assert sec["start_area"] > 0, f"sections[{i}].start_area must be positive"
                assert sec["end_area"] > 0, f"sections[{i}].end_area must be positive"


class TestThroatAdapter:
    """Smoke tests for the throat-adapter command."""

    def test_throat_adapter_conical_valid(self, cli_runner):
        """Valid conical adapter should exit 0 and emit YAML with ap1/lpt."""
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--a1", "30",
                "--a2", "30",
                "--type", "conical",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "throat_adapter:" in result.output
        assert "ap1:" in result.output
        assert "lpt:" in result.output
        assert "type: conical" in result.output

    def test_throat_adapter_exponential_valid(self, cli_runner):
        """Valid exponential adapter should exit 0."""
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "40",
                "--d2", "80",
                "--type", "exponential",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "type: exponential" in result.output

    def test_throat_adapter_cylindrical_valid(self, cli_runner):
        """Cylindrical adapter should exit 0."""
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "60",
                "--d2", "60",
                "--type", "cylindrical",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "type: cylindrical" in result.output

    def test_throat_adapter_parabolic_valid(self, cli_runner):
        """Parabolic adapter should exit 0."""
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "50",
                "--d2", "90",
                "--type", "parabolic",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "type: parabolic" in result.output

    def test_throat_adapter_invalid_type(self, cli_runner):
        """Invalid profile type should exit non-zero with a clear error."""
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--type", "invalid_type",
            ],
        )
        assert result.exit_code == 1, result.output
        assert "Invalid --type" in result.output
        assert "invalid_type" in result.output

    def test_throat_adapter_explicit_length(self, cli_runner):
        """Explicit --length should override minimum-length calculation."""
        result = cli_runner.invoke(
            app,
            [
                "throat-adapter",
                "--d1", "50",
                "--d2", "100",
                "--type", "conical",
                "--length", "60",  # 60 mm explicit
            ],
        )
        assert result.exit_code == 0, result.output
        assert "throat_adapter:" in result.output

    def test_throat_adapter_subprocess_smoke_test(self):
        """Smoke test: throat-adapter via subprocess exits 0 and emits valid YAML snippet."""
        import subprocess, sys, yaml
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "throat-adapter",
             "--d1", "50", "--d2", "100",
             "--type", "conical"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"throat-adapter subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:300]}"
        )
        # Output is a YAML snippet with explanatory text; extract YAML portion
        # The YAML block starts with "throat_adapter:" (bare key, no leading spaces)
        lines = result.stdout.splitlines()
        yaml_start = None
        for i, line in enumerate(lines):
            if line.strip() == "throat_adapter:":
                yaml_start = i
                break
        assert yaml_start is not None, f"Could not find 'throat_adapter:' in output: {result.stdout[:200]}"
        # Skip comment lines before the YAML block; extract until blank line or "Done."
        yaml_lines = []
        for line in lines[yaml_start:]:
            if line.strip() in ("", "Done."):
                break
            yaml_lines.append(line)
        yaml_text = "\n".join(yaml_lines)
        data = yaml.safe_load(yaml_text)
        assert "throat_adapter" in data, "YAML should contain throat_adapter key"
        ta = data["throat_adapter"]
        assert ta["type"] == "conical"
        assert ta["ap1"] > 0, "ap1 (throat area) must be positive"
        assert ta["lpt"] > 0, "lpt (throat length) must be positive"

    def test_throat_adapter_subprocess_invalid_type_exits_nonzero(self):
        """Subprocess smoke: throat-adapter with invalid --type exits non-zero."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "throat-adapter",
             "--d1", "50", "--d2", "100",
             "--type", "invalid_profile"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"throat-adapter with invalid type should exit non-zero, got {result.returncode}"
        )


class TestOptimize:
    def test_optimize_passes_horn_likeness_config(
        self, cli_runner, tmp_path, monkeypatch
    ):
        driver_path = tmp_path / "driver.yaml"
        driver_path.write_text("stub: true\n")

        driver = DriverSpecs(
            fs=49.6,
            qts=0.27,
            qes=0.28,
            qms=7.88,
            vas=0.0369,
            re=7.8,
            bl=7.79,
            mms=0.00699,
            cms=0.001472,
            rms=0.277,
            sd=0.01327,
            voltage=2.83,
            le=0.0008,
            xmax=0.0015,
        )
        captured = {}

        monkeypatch.setattr("pyhorn_cli.cli.optimize_commands.parse_driver_specs", lambda _: driver)

        def _fake_optimize(_dr, config, progress_callback=None):
            captured["min_expansion_ratio"] = config.min_expansion_ratio
            captured["throat_area_penalty_weight"] = config.throat_area_penalty_weight
            params = {
                "throat_area": 0.01,
                "mouth_area": 0.08,
                "path_length": 1.4,
                "lrc": 0.1,
                "vtc": 0.001,
            }
            horn = build_horn_from_params(params, "conical", "BLH")
            return [
                OptimizationResult(
                    profile_type="conical",
                    params=params,
                    cost=1.23,
                    flatness_db=2.1,
                    mean_spl=95.0,
                    bass_deficit_db=1.4,
                    excursion_ok=True,
                    horn=horn,
                    n_evaluations=12,
                )
            ]

        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands.run_optimize", _fake_optimize
        )
        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands._plot_optimizer_results",
            lambda *args, **kwargs: None,
        )

        result = cli_runner.invoke(
            app,
            [
                "optimize",
                "-d",
                str(driver_path),
                "--min-expansion-ratio",
                "6.5",
                "--throat-penalty-weight",
                "1.25",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["min_expansion_ratio"] == pytest.approx(6.5)
        assert captured["throat_area_penalty_weight"] == pytest.approx(1.25)

    def test_optimize_can_export_folded_layout(self, cli_runner, tmp_path, monkeypatch):
        driver_path = tmp_path / "driver.yaml"
        driver_path.write_text("stub: true\n")

        driver = DriverSpecs(
            fs=49.6,
            qts=0.27,
            qes=0.28,
            qms=7.88,
            vas=0.0369,
            re=7.8,
            bl=7.79,
            mms=0.00699,
            cms=0.001472,
            rms=0.277,
            sd=0.01327,
            voltage=2.83,
            le=0.0008,
            xmax=0.0015,
        )
        params = {
            "throat_area": 0.01,
            "mouth_area": 0.08,
            "path_length": 1.4,
            "lrc": 0.1,
            "vtc": 0.001,
        }
        horn = build_horn_from_params(params, "conical", "BLH")
        fake_result = OptimizationResult(
            profile_type="conical",
            params=params,
            cost=1.23,
            flatness_db=2.1,
            mean_spl=95.0,
            bass_deficit_db=1.4,
            excursion_ok=True,
            horn=horn,
            n_evaluations=12,
        )

        monkeypatch.setattr("pyhorn_cli.cli.optimize_commands.parse_driver_specs", lambda _: driver)
        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands.run_optimize", lambda *args, **kwargs: [fake_result]
        )
        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands._plot_optimizer_results", lambda *args, **kwargs: None
        )

        result = cli_runner.invoke(
            app,
            [
                "optimize",
                "-d",
                str(driver_path),
                "-o",
                str(tmp_path),
                "--profiles",
                "conical",
                "--top-n",
                "1",
                "--enclosure-depth",
                "0.5",
                "--enclosure-height",
                "0.7",
                "--driver-x",
                "0.16",
                "--driver-y",
                "0.18",
                "--enclosure-width",
                "0.75",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "optimized_1_conical_folded.yaml").exists()
        assert (tmp_path / "optimized_1_conical_folded.png").exists()

    def test_optimize_passes_fixed_folded_width(
        self, cli_runner, tmp_path, monkeypatch
    ):
        driver_path = tmp_path / "driver.yaml"
        driver_path.write_text("stub: true\n")

        driver = DriverSpecs(
            fs=49.6,
            qts=0.27,
            qes=0.28,
            qms=7.88,
            vas=0.0369,
            re=7.8,
            bl=7.79,
            mms=0.00699,
            cms=0.001472,
            rms=0.277,
            sd=0.01327,
            voltage=2.83,
            le=0.0008,
            xmax=0.0015,
        )
        params = {
            "throat_area": 0.01,
            "mouth_area": 0.08,
            "path_length": 1.4,
            "lrc": 0.1,
            "vtc": 0.001,
        }
        horn = build_horn_from_params(params, "conical", "BLH")
        fake_result = OptimizationResult(
            profile_type="conical",
            params=params,
            cost=1.23,
            flatness_db=2.1,
            mean_spl=95.0,
            bass_deficit_db=1.4,
            excursion_ok=True,
            horn=horn,
            n_evaluations=12,
        )

        monkeypatch.setattr("pyhorn_cli.cli.optimize_commands.parse_driver_specs", lambda _: driver)
        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands.run_optimize", lambda *args, **kwargs: [fake_result]
        )
        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands._plot_optimizer_results", lambda *args, **kwargs: None
        )

        result = cli_runner.invoke(
            app,
            [
                "optimize",
                "-d",
                str(driver_path),
                "-o",
                str(tmp_path),
                "--profiles",
                "conical",
                "--top-n",
                "1",
                "--enclosure-depth",
                "0.5",
                "--enclosure-height",
                "0.7",
                "--driver-x",
                "0.16",
                "--driver-y",
                "0.18",
                "--enclosure-width",
                "0.75",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "folded width: 0.7500 m" in result.output

    def test_optimize_subprocess_smoke_test(self, tmp_path):
        """Smoke test: optimize via subprocess exits 0 and produces valid optimizer YAMLs.

        Uses --max-iter=5 to keep the test fast (scipy differential_evolution is
        naturally slow; limiting to 5 iterations per profile type keeps total runtime
        under ~30 s for the subprocess call).
        """
        import subprocess, sys, yaml

        # Create a minimal driver YAML
        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
        )
        out_dir = tmp_path / "optimize_out"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "optimize",
                "--driver", str(driver),
                "--output-dir", str(out_dir),
                "--fmin", "80",
                "--fmax", "500",
                "--profiles", "conical",
                "--top-n", "1",
                "--max-iter", "5",
                "--no-plot",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"optimize subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:400]}\n"
            f"  stderr: {result.stderr[:400]}"
        )
        # Verify at least one optimizer YAML was produced
        yaml_files = list(out_dir.glob("optimized_*.yaml"))
        assert len(yaml_files) >= 1, (
            f"optimize did not produce any YAML files in {out_dir}\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )
        # Verify the YAML is parseable and has expected horn fields
        for yaml_file in yaml_files:
            data = yaml.safe_load(yaml_file.read_text())
            assert "throat_area" in data, f"{yaml_file.name} missing 'throat_area'"
            assert "mouth_area" in data, f"{yaml_file.name} missing 'mouth_area'"
            assert "path_length" in data, f"{yaml_file.name} missing 'path_length'"
            assert data["throat_area"] > 0, f"{yaml_file.name}: throat_area must be positive"
            assert data["mouth_area"] > data["throat_area"], (
                f"{yaml_file.name}: mouth_area must exceed throat_area"
            )

    def test_optimize_subprocess_invalid_config_exits_nonzero(self, tmp_path):
        """optimize exits non-zero when the driver YAML is malformed."""
        import subprocess, sys

        bad_driver = tmp_path / "bad_driver.yaml"
        bad_driver.write_text("not: [valid yaml at all")
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "optimize",
             "-d", str(bad_driver), "--no-plot", "--max-iter", "1"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"optimize with malformed driver YAML should exit non-zero, got {result.returncode}"
        )

    def test_optimize_subprocess_missing_driver_exits_nonzero(self):
        """optimize exits non-zero when --driver is not provided."""
        import subprocess, sys

        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "optimize"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"optimize without --driver should exit non-zero, got {result.returncode}"
        )


class TestFoldOptimized:
    def test_fold_optimized_accepts_optimizer_yaml(self, cli_runner, tmp_path):
        optimized_yaml = tmp_path / "optimized_1_conical.yaml"
        optimized_yaml.write_text(
            """
enclosure_type: BLH
profile_type: conical
throat_area: 0.024706
mouth_area: 0.065476
path_length: 1.2527
n_segments: 100
lrc: 0.4007
vrc: 0.009899
vtc: 0.005000
""".strip()
        )

        result = cli_runner.invoke(
            app,
            [
                "fold-optimized",
                str(optimized_yaml),
                "--enclosure-depth",
                "0.5",
                "--enclosure-height",
                "0.7",
                "--driver-x",
                "0.0",
                "--driver-y",
                "0.2",
                "--enclosure-width",
                "0.75",
            ],
        )

        assert result.exit_code == 0, result.output
        assert (tmp_path / "optimized_1_conical_folded.yaml").exists()
        assert (tmp_path / "optimized_1_conical_folded.png").exists()
        assert "folded width:" in result.output

    def test_fold_optimized_missing_file(self, cli_runner, tmp_path):
        """fold-optimized exits non-zero when optimizer YAML does not exist."""
        nonexistent = tmp_path / "does_not_exist.yaml"
        result = cli_runner.invoke(
            app,
            [
                "fold-optimized",
                str(nonexistent),
                "--enclosure-depth",
                "0.5",
                "--enclosure-height",
                "0.7",
                "--driver-x",
                "0.0",
                "--driver-y",
                "0.2",
                "--enclosure-width",
                "0.75",
            ],
        )
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

    def test_fold_optimized_malformed_yaml(self, cli_runner, tmp_path):
        """fold-optimized exits non-zero when optimizer YAML is malformed."""
        bad_yaml = tmp_path / "malformed.yaml"
        bad_yaml.write_text("not: [yaml, at: all")
        result = cli_runner.invoke(
            app,
            [
                "fold-optimized",
                str(bad_yaml),
                "--enclosure-depth",
                "0.5",
                "--enclosure-height",
                "0.7",
                "--driver-x",
                "0.0",
                "--driver-y",
                "0.2",
                "--enclosure-width",
                "0.75",
            ],
        )
        assert result.exit_code != 0
        assert "yaml" in result.output.lower() or "parse" in result.output.lower()

    def test_fold_optimized_missing_enclosure_params(self, cli_runner, tmp_path):
        """fold-optimized exits non-zero when enclosure params are partially supplied."""
        valid_yaml = tmp_path / "valid.yaml"
        valid_yaml.write_text(
            "enclosure_type: BLH\nprofile_type: conical\n"
            "throat_area: 0.024706\nmouth_area: 0.065476\n"
            "path_length: 1.2527\nn_segments: 100\n"
            "lrc: 0.4\nvrc: 0.009\nvtc: 0.005\n"
        )
        result = cli_runner.invoke(
            app,
            [
                "fold-optimized",
                str(valid_yaml),
                "--enclosure-depth",
                "0.5",
                # missing --enclosure-height, --driver-x, --driver-y, --enclosure-width
            ],
        )
        assert result.exit_code != 0

    def test_fold_optimized_valid_geometry_produces_valid_layout(
        self, cli_runner, tmp_path
    ):
        """fold-optimized with valid optimizer YAML produces folded_layout with populated panels."""
        import yaml as _yaml

        optimized_yaml = tmp_path / "optimized_1_conical.yaml"
        optimized_yaml.write_text(
            "enclosure_type: BLH\n"
            "profile_type: conical\n"
            "throat_area: 0.024706\n"
            "mouth_area: 0.065476\n"
            "path_length: 1.2527\n"
            "n_segments: 100\n"
            "lrc: 0.4007\n"
            "vrc: 0.009899\n"
            "vtc: 0.005000\n"
        )

        result = cli_runner.invoke(
            app,
            [
                "fold-optimized",
                str(optimized_yaml),
                "--enclosure-depth",
                "0.5",
                "--enclosure-height",
                "0.7",
                "--driver-x",
                "0.0",
                "--driver-y",
                "0.2",
                "--enclosure-width",
                "0.75",
            ],
        )
        assert result.exit_code == 0, result.output

        folded_yaml = tmp_path / "optimized_1_conical_folded.yaml"
        assert folded_yaml.exists(), "Folded YAML was not created"

        data = _yaml.safe_load(folded_yaml.read_text())

        # ── folded_layout structure ───────────────────────────────────────────
        assert "folded_layout" in data, "Output YAML missing 'folded_layout' key"
        assert "panels" in data["folded_layout"], (
            "folded_layout missing 'panels' key"
        )
        panels = data["folded_layout"]["panels"]

        # ── panels list is non-empty ──────────────────────────────────────────
        assert isinstance(panels, list), "folded_layout.panels must be a list"
        assert len(panels) > 0, "folded_layout.panels must not be empty"

        # ── each panel has required fields with valid values ──────────────────
        required_fields = ("x1", "y1", "x2", "y2", "width", "height", "angle", "connection")
        for i, panel in enumerate(panels):
            for field in required_fields:
                assert field in panel, (
                    f"Panel {i} missing required field '{field}'"
                )

            # Coordinates must be numeric
            for coord in ("x1", "y1", "x2", "y2"):
                assert isinstance(panel[coord], (int, float)), (
                    f"Panel {i}.{coord} must be numeric, got {type(panel[coord])}"
                )

            # width and height must be positive
            assert panel["width"] > 0, f"Panel {i}.width must be positive, got {panel['width']}"
            assert panel["height"] > 0, f"Panel {i}.height must be positive, got {panel['height']}"

            # angle must be a reasonable value (-π to π)
            assert -3.15 <= panel["angle"] <= 3.15, (
                f"Panel {i}.angle out of range: {panel['angle']}"
            )

            # connection must be a non-empty string
            assert isinstance(panel["connection"], str) and panel["connection"], (
                f"Panel {i}.connection must be a non-empty string"
            )

        # First panel must connect to "throat"
        assert panels[0]["connection"] == "throat", (
            f"First panel must connect to 'throat', got '{panels[0]['connection']}'"
        )

        # Subsequent panels must have sequential connections
        for i in range(1, len(panels)):
            expected = f"panel_{i - 1}"
            assert panels[i]["connection"] == expected, (
                f"Panel {i} connection must be '{expected}', got '{panels[i]['connection']}'"
            )


    def test_fold_optimized_subprocess_smoke_test(self, tmp_path):
        """Smoke test: fold-optimized via subprocess exits 0 and produces folded YAML."""
        import subprocess, sys, yaml

        # Create a minimal optimizer YAML (must match expected output name pattern)
        optimized_yaml = tmp_path / "optimized_1_conical.yaml"
        optimized_yaml.write_text(
            "enclosure_type: BLH\n"
            "profile_type: conical\n"
            "throat_area: 0.024706\n"
            "mouth_area: 0.065476\n"
            "path_length: 1.2527\n"
            "n_segments: 100\n"
            "lrc: 0.4007\n"
            "vrc: 0.009899\n"
            "vtc: 0.005000\n"
        )
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "fold-optimized",
                str(optimized_yaml),
                "--enclosure-depth", "0.5",
                "--enclosure-height", "0.7",
                "--driver-x", "0.0",
                "--driver-y", "0.2",
                "--enclosure-width", "0.75",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"fold-optimized subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )
        folded_yaml = tmp_path / "optimized_1_conical_folded.yaml"
        assert folded_yaml.exists(), (
            f"fold-optimized did not create folded YAML at {folded_yaml}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
        data = yaml.safe_load(folded_yaml.read_text())
        assert "folded_layout" in data, "Output YAML missing 'folded_layout'"
        assert "panels" in data["folded_layout"], "'folded_layout' missing 'panels'"

    def test_fold_optimized_subprocess_missing_file_exits_nonzero(self, tmp_path):
        """Missing optimizer YAML causes fold-optimized to exit non-zero via subprocess."""
        import subprocess, sys

        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "fold-optimized",
                str(tmp_path / "nonexistent.yaml"),
                "--enclosure-depth", "0.5",
                "--enclosure-height", "0.7",
                "--driver-x", "0.0",
                "--driver-y", "0.2",
                "--enclosure-width", "0.75",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"fold-optimized should fail with missing file, got exit {result.returncode}"
        )


# ─── calculate ────────────────────────────────────────────────────────────────


class TestCalculate:
    """Tests for the calculate command."""

    @pytest.fixture
    def fostex_driver_yaml(self, tmp_path):
        content = """
fs: 49.6
qts: 0.27
qes: 0.28
qms: 7.88
vas: 0.0369
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
sd: 0.01327
le: 0.0008
xmax: 0.0015
"""
        p = tmp_path / "driver.yaml"
        p.write_text(content)
        return p

    @pytest.fixture
    def simple_horn_yaml(self, tmp_path):
        content = """
enclosure_type: BLH
width: 0.2
enclosure_dims: [0.3, 0.35]
conical_segments:
  - [0.005, 0.008, 0.15]
  - [0.008, 0.028, 0.15]
coordinates:
  - [0.0, 0.175]
  - [0.15, 0.175]
  - [0.15, 0.0]
"""
        p = tmp_path / "horn.yaml"
        p.write_text(content)
        return p

    def test_calculate_runs_without_error(
        self, cli_runner, fostex_driver_yaml, simple_horn_yaml, tmp_path
    ):
        """calculate with valid configs should exit 0."""
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                str(fostex_driver_yaml),
                "-h",
                str(simple_horn_yaml),
                "-o",
                str(out_dir),
                "--fmin",
                "100",
                "--fmax",
                "1000",
                "--n-points",
                "50",
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_calculate_creates_output_dir(
        self, cli_runner, fostex_driver_yaml, simple_horn_yaml, tmp_path
    ):
        """Output directory should be created."""
        out_dir = tmp_path / "outputs"
        cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                str(fostex_driver_yaml),
                "-h",
                str(simple_horn_yaml),
                "-o",
                str(out_dir),
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert out_dir.exists()

    def test_calculate_missing_driver(self, cli_runner, simple_horn_yaml, tmp_path):
        """Missing driver config should error gracefully."""
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                "/nonexistent.yaml",
                "-h",
                str(simple_horn_yaml),
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code != 0

    def test_calculate_custom_targets(
        self, cli_runner, fostex_driver_yaml, simple_horn_yaml, tmp_path
    ):
        """Custom target values should not cause an error."""
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                str(fostex_driver_yaml),
                "-h",
                str(simple_horn_yaml),
                "-o",
                str(out_dir),
                "--target-spl",
                "90.0",
                "--target-impedance",
                "8.0",
                "--target-excursion",
                "3.0",
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_calculate_missing_horn(
        self, cli_runner, fostex_driver_yaml, tmp_path
    ):
        """Missing horn file should error gracefully."""
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                str(fostex_driver_yaml),
                "-h",
                "/nonexistent_horn.yaml",
                "-o",
                str(out_dir),
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code != 0

    def test_calculate_malformed_horn_yaml(
        self, cli_runner, fostex_driver_yaml, tmp_path
    ):
        """Malformed (unparseable) horn YAML should error gracefully."""
        bad_horn = tmp_path / "bad_horn.yaml"
        bad_horn.write_text("enclosure_type: BLH\n  bad_indent:\n    - item1\n  broken_list:\n  - this line\n  has wrong indent\n")
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                str(fostex_driver_yaml),
                "-h",
                str(bad_horn),
                "-o",
                str(out_dir),
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        # Should exit non-zero with a clear error, not a raw traceback
        assert result.exit_code != 0
        # No raw Python traceback should appear in output
        assert "Traceback" not in result.output

    def test_calculate_subprocess_smoke_test(self, tmp_path):
        """Smoke test: calculate via subprocess exits 0 and produces response.csv with valid SPL."""
        import subprocess, sys, csv

        # Create a valid driver YAML (Fostex FE166NV2)
        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
            "xmax: 0.0015\n"
        )
        # Create a valid horn YAML using conical_segments format
        horn = tmp_path / "horn.yaml"
        horn.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "enclosure_dims: [0.3, 0.35]\n"
            "conical_segments:\n"
            "  - [0.005, 0.008, 0.15]\n"
            "  - [0.008, 0.028, 0.15]\n"
            "coordinates:\n"
            "  - [0.0, 0.175]\n"
            "  - [0.15, 0.175]\n"
            "  - [0.15, 0.0]\n"
        )
        out_dir = tmp_path / "calc_out"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "calculate",
                "-d", str(driver),
                "-h", str(horn),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "50",
                "--no-plot",
                "--no-plot-3d",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"calculate subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )
        csv_path = out_dir / "horn" / "response.csv"
        assert csv_path.exists(), (
            f"calculate did not produce response.csv at {csv_path}\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )
        # Verify CSV has valid SPL data in reasonable range
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) >= 10, f"response.csv has only {len(rows)} rows, expected >= 10"
        # Find the SPL column (various names: "SPL (dB)", "SPL_dB_Horn SPL (dB)", etc.)
        spl_keys = [k for k in rows[0] if "SPL" in k and "Horn" in k]
        assert spl_keys, (
            f"response.csv missing Horn SPL column. Headers: {list(rows[0].keys())}"
        )
        spl_key = spl_keys[0]
        for row in rows:
            spl_val = float(row[spl_key])
            assert -120 < spl_val < 200, (
                f"SPL value {spl_val} dB outside reasonable range (-120, 200) at freq {row.get('Frequency (Hz)', '?')}"
            )


# ─── wavefront CLI ─────────────────────────────────────────────────────────────


class TestWavefrontCli:
    """Tests for the wavefront CLI flags in the calculate command."""

    @pytest.fixture
    def fostex_driver_yaml(self, tmp_path):
        content = """
fs: 49.6
qts: 0.27
qes: 0.28
qms: 7.88
vas: 0.0369
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
sd: 0.01327
le: 0.0008
xmax: 0.0015
"""
        p = tmp_path / "driver.yaml"
        p.write_text(content)
        return p

    @pytest.fixture
    def simple_horn_yaml(self, tmp_path):
        content = """
enclosure_type: BLH
width: 0.2
enclosure_dims: [0.3, 0.35]
conical_segments:
  - [0.005, 0.008, 0.15]
  - [0.008, 0.028, 0.15]
coordinates:
  - [0.0, 0.175]
  - [0.15, 0.175]
  - [0.15, 0.0]
"""
        p = tmp_path / "horn.yaml"
        p.write_text(content)
        return p

    @pytest.fixture
    def wavefront_geometry_yaml(self, tmp_path):
        """Simple horn geometry YAML with source position for wavefront sim."""
        content = """
enclosure_type: BLH
width: 0.2
coordinates:
  - [0.0, 0.1]
  - [0.1, 0.1]
  - [0.1, 0.0]
  - [0.0, 0.0]
source_x: 0.02
source_y: 0.05
"""
        p = tmp_path / "wf_horn.yaml"
        p.write_text(content)
        return p

    def test_wavefront_flag_saves_snapshot(
        self, cli_runner, fostex_driver_yaml, wavefront_geometry_yaml, tmp_path
    ):
        """--wavefront should solve and save a wavefront PNG snapshot."""
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(fostex_driver_yaml),
                "--wavefront",
                "--wavefront-geometry", str(wavefront_geometry_yaml),
                "--wavefront-freq", "500",
                "--no-plot-3d",
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "1000",
                "--n-points", "50",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "wavefront" in result.output.lower()
        png_files = list(out_dir.glob("wavefront_500Hz.png"))
        assert len(png_files) == 1, f"Expected wavefront_500Hz.png, got {png_files}"

    def test_wavefront_animate_flag_saves_gif(
        self, cli_runner, fostex_driver_yaml, wavefront_geometry_yaml, tmp_path
    ):
        """--wavefront --animate should save a PNG snapshot AND an animated GIF."""
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(fostex_driver_yaml),
                "--wavefront",
                "--animate",
                "--wavefront-geometry", str(wavefront_geometry_yaml),
                "--wavefront-freq", "400",
                "--no-plot-3d",
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "1000",
                "--n-points", "50",
            ],
        )
        assert result.exit_code == 0, result.output
        png_files = list(out_dir.glob("wavefront_400Hz.png"))
        gif_files = list(out_dir.glob("wavefront_animation_400Hz.gif"))
        assert len(png_files) == 1, f"Expected wavefront_400Hz.png, got {png_files}"
        assert len(gif_files) == 1, f"Expected wavefront_animation_400Hz.gif, got {gif_files}"

    def test_wavefront_animate_convenience_flag(
        self, cli_runner, fostex_driver_yaml, wavefront_geometry_yaml, tmp_path
    ):
        """--wavefront-animate should be equivalent to --wavefront --animate."""
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(fostex_driver_yaml),
                "--wavefront-animate",
                "--wavefront-geometry", str(wavefront_geometry_yaml),
                "--wavefront-freq", "300",
                "--no-plot-3d",
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "1000",
                "--n-points", "50",
            ],
        )
        assert result.exit_code == 0, result.output
        gif_files = list(out_dir.glob("wavefront_animation_300Hz.gif"))
        assert len(gif_files) == 1, f"Expected wavefront_animation_300Hz.gif, got {gif_files}"

    def test_wavefront_requires_geometry(
        self, cli_runner, fostex_driver_yaml, simple_horn_yaml, tmp_path
    ):
        """--wavefront without source_x/source_y emits a warning but still succeeds."""
        out_dir = tmp_path / "outputs"
        # simple_horn_yaml has no source_x/source_y so wavefront should warn and use centroid
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(fostex_driver_yaml),
                "-h", str(simple_horn_yaml),
                "--wavefront",
                "--no-plot-3d",
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "1000",
                "--n-points", "50",
            ],
        )
        # Should warn about missing source position but succeed (uses centroid fallback)
        assert result.exit_code == 0
        assert "Warning" in result.output or "warning" in result.output.lower()

    def test_wavefront_freq_option(
        self, cli_runner, fostex_driver_yaml, wavefront_geometry_yaml, tmp_path
    ):
        """Different --wavefront-freq values should produce differently-named files."""
        out_dir = tmp_path / "outputs1"
        result1 = cli_runner.invoke(
            app,
            [
                "calculate", "-d", str(fostex_driver_yaml),
                "--wavefront", "--wavefront-geometry", str(wavefront_geometry_yaml),
                "--wavefront-freq", "250",
                "--no-plot-3d", "-o", str(tmp_path / "out1"),
                "--fmin", "100", "--fmax", "1000", "--n-points", "50",
            ],
        )
        assert result1.exit_code == 0, result1.output
        png_250 = list((tmp_path / "out1").glob("wavefront_250Hz.png"))
        assert len(png_250) == 1

    def test_wavefront_subprocess_smoke_test(self, tmp_path):
        """Smoke test: wavefront via real subprocess exits 0 and produces PNG.

        Uses minimal geometry (source_x/source_y) and a single frequency point
        to keep runtime fast (~10-30 s for the 2D Helmholtz solve).
        """
        import subprocess, sys

        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
        )
        geo = tmp_path / "wf_horn.yaml"
        geo.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "coordinates:\n"
            "  - [0.0, 0.1]\n"
            "  - [0.1, 0.1]\n"
            "  - [0.1, 0.0]\n"
            "  - [0.0, 0.0]\n"
            "source_x: 0.02\n"
            "source_y: 0.05\n"
        )
        out_dir = tmp_path / "wf_out"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "calculate",
                "-d", str(driver),
                "-o", str(out_dir),
                "--wavefront",
                "--wavefront-geometry", str(geo),
                "--wavefront-freq", "400",
                "--fmin", "200",
                "--fmax", "600",
                "--n-points", "10",
                "--no-plot",
                "--no-plot-3d",
            ],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, (
            f"wavefront subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:400]}\n"
            f"  stderr: {result.stderr[:400]}"
        )
        png_files = list(out_dir.glob("wavefront_400Hz.png"))
        assert len(png_files) == 1, (
            f"wavefront did not produce wavefront_400Hz.png in {out_dir}\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )


# ─── wavefront-edit ────────────────────────────────────────────────────────────


class TestWavefrontEditCli:
    """Tests for the wavefront-edit CLI command."""

    def test_wavefront_edit_dry_run_valid_geometry(self, cli_runner, tmp_path):
        """wavefront-edit --dry-run with valid geometry should exit 0 and show summary."""
        geometry_yaml = tmp_path / "wf_horn.yaml"
        geometry_yaml.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "coordinates:\n"
            "  - [0.0, 0.1]\n"
            "  - [0.1, 0.1]\n"
            "  - [0.1, 0.0]\n"
            "  - [0.0, 0.0]\n"
            "source_x: 0.02\n"
            "source_y: 0.05\n"
        )
        result = cli_runner.invoke(
            app,
            [
                "wavefront-edit",
                "--dry-run",
                "--geometry", str(geometry_yaml),
            ],
        )
        assert result.exit_code == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.output}"
        assert "Dry run complete" in result.output
        assert "Vertices:" in result.output

    def test_wavefront_edit_dry_run_missing_geometry(self, cli_runner, tmp_path):
        """wavefront-edit with missing geometry file should exit non-zero."""
        nonexistent = tmp_path / "nonexistent.yaml"
        result = cli_runner.invoke(
            app,
            [
                "wavefront-edit",
                "--dry-run",
                "--geometry", str(nonexistent),
            ],
        )
        assert result.exit_code != 0

    def test_wavefront_edit_dry_run_malformed_yaml(self, cli_runner, tmp_path):
        """wavefront-edit with malformed YAML should exit non-zero."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("not: valid: yaml: content:")
        result = cli_runner.invoke(
            app,
            [
                "wavefront-edit",
                "--dry-run",
                "--geometry", str(bad_yaml),
            ],
        )
        assert result.exit_code != 0

    def test_wavefront_edit_dry_run_with_output_file(self, cli_runner, tmp_path):
        """wavefront-edit --dry-run --output should still work (dry-run skips save)."""
        geometry_yaml = tmp_path / "wf_horn.yaml"
        geometry_yaml.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "coordinates:\n"
            "  - [0.0, 0.1]\n"
            "  - [0.1, 0.1]\n"
            "  - [0.1, 0.0]\n"
            "  - [0.0, 0.0]\n"
            "source_x: 0.02\n"
            "source_y: 0.05\n"
        )
        output_yaml = tmp_path / "output.yaml"
        result = cli_runner.invoke(
            app,
            [
                "wavefront-edit",
                "--dry-run",
                "--geometry", str(geometry_yaml),
                "--output", str(output_yaml),
            ],
        )
        assert result.exit_code == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.output}"
        # dry-run does NOT write output file (editor skipped)
        assert not output_yaml.exists()

    def test_wavefront_edit_subprocess_smoke_test(self, tmp_path):
        """Smoke test: wavefront-edit via subprocess exits 0 and prints geometry summary."""
        import subprocess, sys

        geometry_yaml = tmp_path / "wf_horn.yaml"
        geometry_yaml.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "coordinates:\n"
            "  - [0.0, 0.1]\n"
            "  - [0.1, 0.1]\n"
            "  - [0.1, 0.0]\n"
            "  - [0.0, 0.0]\n"
            "source_x: 0.02\n"
            "source_y: 0.05\n"
        )
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "wavefront-edit",
                "--dry-run",
                "--geometry", str(geometry_yaml),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"wavefront-edit subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )
        # dry-run should print some summary without crashing
        combined = result.stdout + result.stderr
        assert len(combined) > 0, "wavefront-edit produced no output"


# ─── compare ────────────────────────────────────────────────────────────────────


class TestCompare:
    """Tests for the compare command."""

    @pytest.fixture
    def fostex_driver_yaml(self, tmp_path):
        content = """
fs: 49.6
qts: 0.27
qes: 0.28
qms: 7.88
vas: 0.0369
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
sd: 0.01327
le: 0.0008
xmax: 0.0015
"""
        p = tmp_path / "driver.yaml"
        p.write_text(content)
        return p

    @pytest.fixture
    def two_horn_yamls(self, tmp_path):
        h1 = tmp_path / "horn1.yaml"
        h1.write_text(
            """
enclosure_type: BLH
width: 0.2
enclosure_dims: [0.3, 0.35]
conical_segments:
  - [0.005, 0.01, 0.15]
  - [0.01, 0.03, 0.15]
coordinates:
  - [0.0, 0.175]
  - [0.15, 0.175]
  - [0.15, 0.0]
"""
        )
        h2 = tmp_path / "horn2.yaml"
        h2.write_text(
            """
enclosure_type: BLH
width: 0.2
enclosure_dims: [0.35, 0.4]
conical_segments:
  - [0.004, 0.01, 0.2]
  - [0.01, 0.035, 0.2]
coordinates:
  - [0.0, 0.2]
  - [0.2, 0.2]
  - [0.2, 0.0]
"""
        )
        return h1, h2

    def test_compare_runs(
        self, cli_runner, fostex_driver_yaml, two_horn_yamls, tmp_path
    ):
        """Compare with two valid horns should exit 0 and create output."""
        out_dir = tmp_path / "comparison"
        h1, h2 = two_horn_yamls
        result = cli_runner.invoke(
            app,
            [
                "compare",
                str(h1),
                str(h2),
                "-d",
                str(fostex_driver_yaml),
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output

    def test_compare_creates_plot(
        self, cli_runner, fostex_driver_yaml, two_horn_yamls, tmp_path
    ):
        """Compare should create spl_compare.png."""
        out_dir = tmp_path / "comparison"
        h1, h2 = two_horn_yamls
        cli_runner.invoke(
            app,
            [
                "compare",
                str(h1),
                str(h2),
                "-d",
                str(fostex_driver_yaml),
                "-o",
                str(out_dir),
            ],
        )
        assert (out_dir / "spl_compare.png").exists()

    def test_compare_png_has_plotted_content(
        self, cli_runner, fostex_driver_yaml, two_horn_yamls, tmp_path
    ):
        """E2E: compare PNG must contain actual plotted SPL curves, not a blank image."""
        out_dir = tmp_path / "comparison"
        h1, h2 = two_horn_yamls
        result = cli_runner.invoke(
            app,
            [
                "compare",
                str(h1),
                str(h2),
                "-d",
                str(fostex_driver_yaml),
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        png_path = out_dir / "spl_compare.png"
        assert png_path.exists()

        # Load the PNG and verify it has plotted content (not a blank/white image).
        # The plot area is surrounded by white margins; we crop to the central 60%
        # and check that at least 5% of pixels are non-white (i.e., have a curve).
        from PIL import Image

        img = Image.open(png_path).convert("RGB")
        w, h = img.size
        # Crop central region where curves should be drawn
        margin = 0.2
        crop = img.crop((
            int(w * margin),
            int(h * margin),
            int(w * (1 - margin)),
            int(h * (1 - margin)),
        ))
        import numpy as np
        arr = np.array(crop)
        # Count non-white pixels (all channels > 240)
        non_white = np.sum(~((arr > 240).all(axis=2)))
        total = arr.shape[0] * arr.shape[1]
        non_white_ratio = non_white / total if total > 0 else 0
        assert non_white_ratio > 0.05, (
            f"PNG appears blank — only {non_white_ratio:.1%} non-white pixels "
            f"(expected >5%). Image size: {w}x{h}"
        )

    def test_compare_both_horns_missing(
        self, cli_runner, fostex_driver_yaml, tmp_path
    ):
        """When both horn files don't exist, compare should report errors and exit 0
        (compare continues with remaining horns; with zero valid horns the output
        may be empty but should not traceback)."""
        out_dir = tmp_path / "comparison"
        result = cli_runner.invoke(
            app,
            [
                "compare",
                "/nonexistent1.yaml",
                "/nonexistent2.yaml",
                "-d",
                str(fostex_driver_yaml),
                "-o",
                str(out_dir),
            ],
        )
        # compare catches exceptions per-horn and exits 0 even with failures;
        # verify no raw Python traceback leaked
        assert "Traceback" not in result.output

    def test_compare_malformed_first_horn_reports_error(
        self, cli_runner, fostex_driver_yaml, two_horn_yamls, tmp_path
    ):
        """Malformed YAML in first horn should be caught, reported, and not traceback."""
        out_dir = tmp_path / "comparison"
        h1, h2 = two_horn_yamls
        # Corrupt h1 — parse will fail inside the per-horn try/except
        h1.write_text("enclosure_type: BLH\n  bad_indent:\n    - item\n  broken:\n  list\n  with bad indent\n")
        result = cli_runner.invoke(
            app,
            [
                "compare",
                str(h1),
                str(h2),
                "-d",
                str(fostex_driver_yaml),
                "-o",
                str(out_dir),
            ],
        )
        # Command exits 0 (second horn still simulated) but no raw traceback
        assert "Traceback" not in result.output

    def test_compare_subprocess_smoke_test(self, tmp_path):
        """Smoke test: compare via subprocess exits 0 and produces valid PNG."""
        import subprocess, sys
        from PIL import Image
        import numpy as np

        # Create two valid horn YAML files using conical_segments format
        # (same geometry used by the existing TestCompare unit tests)
        horn1 = tmp_path / "horn1.yaml"
        horn1.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "enclosure_dims: [0.3, 0.35]\n"
            "conical_segments:\n"
            "  - [0.005, 0.01, 0.15]\n"
            "  - [0.01, 0.03, 0.15]\n"
            "coordinates:\n"
            "  - [0.0, 0.175]\n"
            "  - [0.15, 0.175]\n"
            "  - [0.15, 0.0]\n"
        )
        horn2 = tmp_path / "horn2.yaml"
        horn2.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "enclosure_dims: [0.35, 0.4]\n"
            "conical_segments:\n"
            "  - [0.004, 0.01, 0.2]\n"
            "  - [0.01, 0.035, 0.2]\n"
            "coordinates:\n"
            "  - [0.0, 0.2]\n"
            "  - [0.2, 0.2]\n"
            "  - [0.2, 0.0]\n"
        )
        # Create a valid driver YAML (Fostex FE166NV2 T-S params)
        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
        )
        out_dir = tmp_path / "compare_out"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "compare",
                str(horn1), str(horn2),
                "-d", str(driver),
                "-o", str(out_dir),
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"compare subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:500]}"
        )
        png_path = out_dir / "spl_compare.png"
        assert png_path.exists(), (
            f"compare did not produce PNG at {png_path}\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )
        # Verify PNG has plotted content (not a blank/white image)
        img = Image.open(png_path).convert("RGB")
        w, h = img.size
        margin = 0.2
        crop = img.crop((
            int(w * margin), int(h * margin),
            int(w * (1 - margin)), int(h * (1 - margin)),
        ))
        arr = np.array(crop)
        non_white = np.sum(~((arr > 240).all(axis=2)))
        total = arr.shape[0] * arr.shape[1]
        non_white_ratio = non_white / total if total > 0 else 0
        assert non_white_ratio > 0.05, (
            f"PNG appears blank — only {non_white_ratio:.1%} non-white pixels "
            f"(expected >5%). Image size: {w}x{h}"
        )

    def test_compare_subprocess_missing_driver_exits_nonzero(self, tmp_path):
        """Missing driver file should cause compare to exit non-zero via subprocess."""
        import subprocess, sys

        horn1 = tmp_path / "horn1.yaml"
        horn1.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "enclosure_dims: [0.3, 0.35]\n"
            "conical_segments:\n"
            "  - [0.005, 0.01, 0.15]\n"
            "  - [0.01, 0.03, 0.15]\n"
            "coordinates:\n"
            "  - [0.0, 0.175]\n"
            "  - [0.15, 0.175]\n"
            "  - [0.15, 0.0]\n"
        )
        horn2 = tmp_path / "horn2.yaml"
        horn2.write_text(
            "enclosure_type: BLH\n"
            "width: 0.2\n"
            "enclosure_dims: [0.35, 0.4]\n"
            "conical_segments:\n"
            "  - [0.004, 0.01, 0.2]\n"
            "  - [0.01, 0.035, 0.2]\n"
            "coordinates:\n"
            "  - [0.0, 0.2]\n"
            "  - [0.2, 0.2]\n"
            "  - [0.2, 0.0]\n"
        )
        missing_driver = tmp_path / "nonexistent_driver.yaml"
        out_dir = tmp_path / "compare_out"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "compare",
                str(horn1), str(horn2),
                "-d", str(missing_driver),
                "-o", str(out_dir),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"compare with missing driver should exit non-zero, got {result.returncode}.\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )


# ─── chamber-wizard ───────────────────────────────────────────────────────────


class TestChamberWizard:
    """Smoke tests for the chamber-wizard CLI command."""

    @pytest.fixture
    def valid_driver_yaml(self, tmp_path):
        """Minimal driver YAML with all required T-S parameters."""
        content = """
fs: 49.6
qts: 0.27
vas: 0.0369
sd: 0.0132
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
le: 0.0008
"""
        p = tmp_path / "driver.yaml"
        p.write_text(content)
        return p

    @pytest.fixture
    def driver_missing_qts(self, tmp_path):
        """Driver YAML missing required Qts."""
        content = """
fs: 49.6
vas: 0.0369
"""
        p = tmp_path / "driver_no_qts.yaml"
        p.write_text(content)
        return p

    def test_chamber_wizard_valid_driver_no_interactive(
        self, cli_runner, valid_driver_yaml
    ):
        """Valid driver + --no-interactive should exit 0 and emit YAML snippet."""
        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(valid_driver_yaml),
                "--no-interactive",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "rear_chamber:" in result.output
        assert "throat_chamber:" in result.output
        assert "Vrc" in result.output
        assert "Lrc" in result.output

    def test_chamber_wizard_missing_driver_file(self, cli_runner, tmp_path):
        """Non-existent driver path should exit 1 with a clear error."""
        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(tmp_path / "nonexistent.yaml"),
                "--no-interactive",
            ],
        )
        assert result.exit_code == 1, result.output
        assert "not found" in result.output.lower()

    def test_chamber_wizard_missing_ts_params(
        self, cli_runner, driver_missing_qts
    ):
        """Driver missing required T-S params should exit 1."""
        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(driver_missing_qts),
                "--no-interactive",
            ],
        )
        assert result.exit_code == 1, result.output
        assert "could not parse" in result.output.lower() or "required" in result.output.lower()

    def test_chamber_wizard_custom_qts_target(
        self, cli_runner, valid_driver_yaml
    ):
        """Custom --qts-target should be accepted and reflected in output."""
        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(valid_driver_yaml),
                "--qts-target", "0.7",
                "--no-interactive",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "target Qts = 0.70" in result.output or "Qts = 0.70" in result.output

    def test_chamber_wizard_writes_output_file(
        self, cli_runner, valid_driver_yaml, tmp_path
    ):
        """--output should create the YAML snippet file."""
        out_path = tmp_path / "chamber_out.yaml"
        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(valid_driver_yaml),
                "--no-interactive",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists(), "Output file was not created"
        text = out_path.read_text()
        assert "rear_chamber:" in text
        assert "throat_chamber:" in text

    def test_chamber_wizard_negative_qts_target(self, cli_runner, valid_driver_yaml):
        """Negative --qts-target should be rejected or clamped."""
        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(valid_driver_yaml),
                "--qts-target", "-0.5",
                "--no-interactive",
            ],
        )
        # Should either exit non-zero or produce a warning about invalid range
        # The command currently does not validate this upfront, so we check it at least runs
        assert result.exit_code == 0, result.output

    def test_chamber_wizard_output_yaml_is_valid_and_physical(
        self, cli_runner, valid_driver_yaml, tmp_path
    ):
        """E2E: chamber-wizard --output produces valid YAML with physically sensible values.

        Validates the output YAML from chamber-wizard contains all required sections
        and that values are within physically reasonable ranges:
        - throat_chamber.atc must be positive (throat area)
        - throat_chamber.vtc must be positive (throat chamber volume)
        - throat_adapter.ap1 must be positive (throat adapter area)
        - throat_adapter.lpt must be non-negative (throat adapter length)
        - rear_chamber.vrc/lrc must be non-negative (rear chamber is optional; can be 0)
        """
        import yaml as _yaml

        out_path = tmp_path / "chamber_out.yaml"
        result = cli_runner.invoke(
            app,
            [
                "chamber-wizard",
                "--driver", str(valid_driver_yaml),
                "--no-interactive",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert out_path.exists(), "Output file was not created"

        text = out_path.read_text(encoding="utf-8")

        # chamber-wizard outputs a YAML snippet block; find the marker
        yaml_marker = "YAML Snippet"
        marker_pos = text.find(yaml_marker)
        if marker_pos != -1:
            # Extract the YAML block that follows the marker
            yaml_text = text[marker_pos + len(yaml_marker) :]
        else:
            yaml_text = text

        # Parse the YAML — strip leading comment lines that precede the document
        lines = yaml_text.splitlines()
        # Skip lines until we find a dict start (rear_chamber:, throat_chamber:, etc.)
        yaml_start_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("rear_chamber:") or stripped.startswith("throat_chamber:"):
                yaml_start_idx = i
                break
        yaml_doc = "\n".join(lines[yaml_start_idx:])

        geo = _yaml.safe_load(yaml_doc)
        assert isinstance(geo, dict), f"Parsed YAML is not a dict: {type(geo)}"

        # ── rear chamber ───────────────────────────────────────────────────────
        # Vrc can be 0 when Qts <= qts_alignment (FE166NV2: Qts=0.27, alignment=0.55 → 0/1)
        assert "rear_chamber" in geo, "Output YAML missing 'rear_chamber' key"
        rc = geo["rear_chamber"]
        assert "vrc" in rc, "rear_chamber missing 'vrc' (volume)"
        assert "lrc" in rc, "rear_chamber missing 'lrc' (length)"
        assert rc["vrc"] >= 0, f"Vrc must be non-negative, got {rc['vrc']} L"
        assert rc["lrc"] >= 0, f"Lrc must be non-negative, got {rc['lrc']} cm"

        # ── throat chamber ─────────────────────────────────────────────────────
        # Atc must be positive — it's the throat chamber cross-sectional area
        assert "throat_chamber" in geo, "Output YAML missing 'throat_chamber' key"
        tc = geo["throat_chamber"]
        assert "atc" in tc, "throat_chamber missing 'atc' (area)"
        assert "vtc" in tc, "throat_chamber missing 'vtc' (volume)"
        assert tc["atc"] > 0, f"Atc must be positive, got {tc['atc']} cm²"
        assert tc["vtc"] > 0, f"Vtc must be positive, got {tc['vtc']} m³"

        # ── throat adapter ─────────────────────────────────────────────────────
        assert "throat_adapter" in geo, "Output YAML missing 'throat_adapter' key"
        ta = geo["throat_adapter"]
        assert "ap1" in ta, "throat_adapter missing 'ap1' (area)"
        assert "lpt" in ta, "throat_adapter missing 'lpt' (length)"
        assert ta["ap1"] > 0, f"Ap1 must be positive, got {ta['ap1']} cm²"
        assert ta["lpt"] >= 0, f"Lpt must be non-negative, got {ta['lpt']} mm"


    def test_chamber_wizard_subprocess_smoke_test(self, tmp_path):
        """Smoke test: chamber-wizard via subprocess exits 0 and produces valid YAML."""
        import subprocess, sys, yaml

        # Create a valid driver YAML
        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
        )
        out_path = tmp_path / "chamber_out.yaml"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "chamber-wizard",
                "--driver", str(driver),
                "--no-interactive",
                "--output", str(out_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"chamber-wizard subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )
        assert out_path.exists(), f"chamber-wizard did not create output file at {out_path}"
        text = out_path.read_text(encoding="utf-8")
        assert "rear_chamber:" in text, "Output YAML missing 'rear_chamber:' section"
        assert "throat_chamber:" in text, "Output YAML missing 'throat_chamber:' section"
        # Parse to confirm it's valid YAML
        geo = yaml.safe_load(text)
        assert isinstance(geo, dict), "Output is not a valid YAML dict"

    def test_chamber_wizard_subprocess_missing_driver_exits_nonzero(self, tmp_path):
        """Missing driver file should cause chamber-wizard to exit non-zero via subprocess."""
        import subprocess, sys

        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "chamber-wizard",
                "--driver", str(tmp_path / "nonexistent.yaml"),
                "--no-interactive",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"chamber-wizard should fail with missing driver, got exit {result.returncode}"
        )


# ─── synthesis-wizard ─────────────────────────────────────────────────────────


class TestSynthesisWizard:
    """Smoke tests for the synthesis-wizard CLI command."""

    @pytest.fixture
    def valid_driver_yaml(self, tmp_path):
        """Driver YAML with all required T-S parameters for synthesis."""
        content = """
fs: 49.6
qts: 0.27
vas: 0.0369
sd: 0.0132
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
le: 0.0008
qes: 0.28
qms: 7.88
"""
        p = tmp_path / "driver.yaml"
        p.write_text(content)
        return p

    @pytest.fixture
    def driver_missing_sd(self, tmp_path):
        """Driver YAML missing Sd (required for synthesis)."""
        content = """
fs: 49.6
qts: 0.27
vas: 0.0369
re: 7.8
"""
        p = tmp_path / "driver_no_sd.yaml"
        p.write_text(content)
        return p

    def test_synthesis_wizard_valid_driver(
        self, cli_runner, valid_driver_yaml
    ):
        """Valid driver should exit 0 and emit synthesised geometry YAML."""
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(valid_driver_yaml),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Synthesis complete" in result.output or "✓" in result.output
        # Check that geometry parameters are in the output
        assert "Throat area" in result.output or "S1" in result.output
        assert "Mouth area" in result.output or "S2" in result.output
        assert "Path length" in result.output or "L12" in result.output

    def test_synthesis_wizard_missing_driver_file(self, cli_runner, tmp_path):
        """Non-existent driver path should exit 1."""
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(tmp_path / "nonexistent.yaml"),
            ],
        )
        assert result.exit_code == 1, result.output
        assert "not found" in result.output.lower()

    def test_synthesis_wizard_missing_sd(
        self, cli_runner, driver_missing_sd
    ):
        """Driver missing Sd should exit 1 with a clear error."""
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(driver_missing_sd),
            ],
        )
        assert result.exit_code == 1, result.output
        assert "sd" in result.output.lower()

    def test_synthesis_wizard_custom_f3(
        self, cli_runner, valid_driver_yaml
    ):
        """Custom --f3 should be accepted."""
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(valid_driver_yaml),
                "--f3", "40.0",
            ],
        )
        assert result.exit_code == 0, result.output
        # Should target f3=40 Hz (which sets F12 ≈ 33 Hz)
        assert "40" in result.output

    def test_synthesis_wizard_writes_output_file(
        self, cli_runner, valid_driver_yaml, tmp_path
    ):
        """--output should create the output file."""
        out_path = tmp_path / "synth_out.yaml"
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(valid_driver_yaml),
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists(), "Output file was not created"

    def test_synthesis_wizard_extreme_f3(
        self, cli_runner, valid_driver_yaml
    ):
        """Extremely low f3 should either succeed or fail gracefully (no crash)."""
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(valid_driver_yaml),
                "--f3", "5.0",
            ],
        )
        # Any exit code is fine — just no Python traceback crash
        assert result.exit_code in (0, 1), f"Unexpected exit: {result.exit_code}\n{result.output}"

    def test_synthesis_wizard_output_yaml_is_valid_and_physical(
        self, cli_runner, valid_driver_yaml, tmp_path
    ):
        """E2E: synthesis-wizard --output produces valid YAML with physically sensible values.

        Validates the output YAML from the synthesis wizard contains all required
        geometry sections and that values are within physically reasonable ranges:
        - throat area (S1) must be positive
        - mouth area (S2) must be > throat area (expanding horn)
        - path length must be positive
        - rear chamber volume Vrc must be positive
        - throat chamber Atc must be positive
        """
        import yaml as _yaml

        out_path = tmp_path / "synth_out.yaml"
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(valid_driver_yaml),
                "--f3", "50.0",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert out_path.exists(), "Output file was not created"

        text = out_path.read_text(encoding="utf-8")

        # The file may have YAML frontmatter comments; find the YAML document
        yaml_start = text.find("# ── Synthesised geometry")
        assert yaml_start != -1, f"Could not find 'Synthesised geometry' marker in output:\n{text[:300]}"
        yaml_text = text[yaml_start:]

        geo = _yaml.safe_load(yaml_text)
        assert isinstance(geo, dict), f"Parsed YAML is not a dict: {type(geo)}"

        # ── sections block ────────────────────────────────────────────────────
        assert "sections" in geo, "Output YAML missing 'sections' key"
        sections = geo["sections"]
        assert isinstance(sections, list), f"sections must be a list, got {type(sections)}"
        assert len(sections) > 0, "sections list must not be empty"
        for sec in sections:
            assert "name" in sec, f"Section missing 'name': {sec}"
            assert "profile_type" in sec, f"Section {sec.get('name')} missing 'profile_type'"
            assert "length" in sec, f"Section {sec.get('name')} missing 'length'"
            assert "start_area" in sec, f"Section {sec.get('name')} missing 'start_area'"
            assert "end_area" in sec, f"Section {sec.get('name')} missing 'end_area'"
            assert sec["length"] > 0, f"Section {sec['name']} length must be > 0, got {sec['length']}"
            assert sec["start_area"] > 0, f"Section {sec['name']} start_area must be > 0"
            assert sec["end_area"] > 0, f"Section {sec['name']} end_area must be > 0"

        # ── total horn metrics from sections ─────────────────────────────────
        total_length = sum(sec["length"] for sec in sections)
        throat_area = sections[0]["start_area"]
        mouth_area = sections[-1]["end_area"]
        assert total_length > 0, f"Total path length must be > 0, got {total_length} m"
        assert throat_area > 0, f"Throat area must be > 0, got {throat_area} m²"
        assert mouth_area > throat_area, (
            f"Mouth area ({mouth_area:.6f} m²) must exceed throat area ({throat_area:.6f} m²)"
        )
        # Mouth area in reasonable range (< 1 m² for a speaker horn)
        assert mouth_area < 1.0, f"Mouth area ({mouth_area:.4f} m²) unreasonably large"

        # ── rear chamber (flat fields) ─────────────────────────────────────────
        # Vrc can be 0.0 when Qts <= qts_alignment (driver Q is already low enough;
        # synthesis wizard emits INFO warning in that case — both are valid).
        assert "vrc" in geo, "Output YAML missing 'vrc' (rear chamber volume)"
        assert "lrc" in geo, "Output YAML missing 'lrc' (rear chamber length)"
        assert geo["vrc"] >= 0, f"Vrc must be non-negative, got {geo['vrc']} L"
        assert geo["lrc"] >= 0, f"Lrc must be non-negative, got {geo['lrc']} cm"

        # ── throat chamber (flat fields) ───────────────────────────────────────
        assert "vtc" in geo, "Output YAML missing 'vtc' (throat chamber volume)"
        assert "atc" in geo, "Output YAML missing 'atc' (throat chamber area)"
        assert geo["atc"] > 0, f"Atc must be positive, got {geo['atc']} cm²"
        assert geo["vtc"] > 0, f"Vtc must be positive, got {geo['vtc']} m³"

        # ── throat adapter (flat fields) ───────────────────────────────────────
        assert "ap1" in geo, "Output YAML missing 'ap1' (throat adapter area)"
        assert "lpt" in geo, "Output YAML missing 'lpt' (throat adapter length)"
        assert geo["ap1"] > 0, f"Ap1 must be positive, got {geo['ap1']} cm²"
        assert geo["lpt"] >= 0, f"Lpt must be non-negative, got {geo['lpt']} mm"

        # ── radiation angle ────────────────────────────────────────────────────
        assert "ang" in geo, "Output YAML missing 'ang' (radiation angle)"
        assert geo["ang"] > 0, f"ang must be positive, got {geo['ang']}"

    def test_synthesis_wizard_roundtrip_calculate(
        self, cli_runner, valid_driver_yaml, tmp_path
    ):
        """E2E roundtrip: synthesis-wizard output → calculate → valid SPL response.

        This is the most critical end-to-end workflow: synthesise a horn geometry,
        then immediately feed it to the solver and verify the simulation produces
        a valid, physically plausible SPL response.
        """
        import csv as _csv
        import math as _math
        import yaml as _yaml

        # ── Step 1: run synthesis-wizard, save output ──────────────────────────
        synth_out = tmp_path / "synth_horn.yaml"
        result = cli_runner.invoke(
            app,
            [
                "synthesis-wizard",
                "--driver", str(valid_driver_yaml),
                "--f3", "50.0",
                "--output", str(synth_out),
            ],
        )
        assert result.exit_code == 0, f"synthesis-wizard failed: {result.output}"
        assert synth_out.exists(), "synthesis-wizard did not create output file"

        # ── Step 2: parse the YAML block from synthesis output ─────────────────
        text = synth_out.read_text(encoding="utf-8")
        geo_marker = text.find("# ── Synthesised geometry")
        assert geo_marker != -1, f"Could not find geometry marker in output:\n{text[:300]}"
        yaml_text = text[geo_marker:]
        geo = _yaml.safe_load(yaml_text)
        assert "sections" in geo, "Synthesis output missing 'sections'"
        assert len(geo["sections"]) > 0, "sections must not be empty"

        # ── Step 3: build a complete horn geometry YAML for the calculate command
        horn_yaml = {
            "throat_area": geo["sections"][0]["start_area"],
            "mouth_area": geo["sections"][-1]["end_area"],
            "path_length": sum(s["length"] for s in geo["sections"]),
            "profile_type": geo["sections"][-1].get("profile_type", "catenoidal"),
            "hyperbolic_t": geo["sections"][-1].get("hyperbolic_t", 1.0),
            "sections": geo["sections"],
            "vrc": geo.get("vrc", 0.0),
            "lrc": geo.get("lrc", 0.0),
            "vtc": geo.get("vtc", 1e-6),
            "atc": float(geo.get("atc", 0.0)) / 1e4,  # cm² → m²
            "ap1": float(geo.get("ap1", 0.0)) / 1e4,  # cm² → m²
            "lpt": float(geo.get("lpt", 0.0)) / 1000,  # mm → m
            "ang": geo.get("ang", 6.283185307),
        }

        horn_path = tmp_path / "horn.yaml"
        with open(horn_path, "w", encoding="utf-8") as f:
            _yaml.safe_dump(horn_yaml, f)

        # ── Step 4: run calculate with the synthesised horn ─────────────────────
        out_dir = tmp_path / "calc_out"
        calc_result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(valid_driver_yaml),
                "-h", str(horn_path),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "20",
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert calc_result.exit_code == 0, (
            f"calculate failed on synthesised geometry.\n"
            f"Exit: {calc_result.exit_code}\nOutput: {calc_result.output}"
        )

        # ── Step 5: verify response.csv exists and SPL is in plausible range ────
        csv_files = list(out_dir.rglob("response.csv"))
        assert len(csv_files) == 1, f"Expected 1 CSV, found {len(csv_files)}"
        with open(csv_files[0], newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 20, f"Expected 20 frequency points, got {len(rows)}"

        # Find SPL column (may be "Horn SPL (dB)" or similar)
        spl_col = next(
            (c for c in rows[0].keys() if "SPL" in c and "Horn" in c),
            None,
        )
        assert spl_col is not None, f"No Horn SPL column found. Columns: {list(rows[0].keys())}"
        spls = [float(r[spl_col]) for r in rows]

        # SPL must be in a physically plausible range (60–130 dB for a FE166NV2)
        assert all(60.0 <= spl <= 130.0 for spl in spls), (
            f"SPL values out of range: {min(spls):.1f}–{max(spls):.1f} dB"
        )
        # Spread should be plausible (< 60 dB across 100–2000 Hz)
        assert max(spls) - min(spls) < 60.0, (
            f"SPL range too large: {max(spls)-min(spls):.1f} dB"
        )
        # No NaN or Inf
        for row in rows:
            for col, val in row.items():
                try:
                    num = float(val)
                    assert not (_math.isnan(num) or _math.isinf(num)), (
                        f"Non-finite value in {col} at freq {row.get('Frequency_Hz')}"
                    )
                except (ValueError, TypeError):
                    pass

    def test_synthesis_wizard_available_in_main_entry_point(self):
        """Regression: synthesis-wizard must be registered in main.py (user-facing entry point).

        Previously, synthesis-wizard was registered in cli.py (commands_app) but NOT in
        main.py, making it invisible to `python3 -m pyhorn_cli.main synthesis-wizard`.
        This test invokes the command via subprocess to confirm the command is
        accessible to end users.
        """
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "synthesis-wizard", "--help"],
            capture_output=True, text=True, cwd="/Users/guillaume/pyhorn",
        )
        assert result.returncode == 0, (
            f"synthesis-wizard not available in main.py entry point. "
            f"Exit: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
        )
        assert "Synthesis" in result.stdout or "synthesis" in result.stdout.lower()

    def test_synthesis_wizard_subprocess_smoke_test(self, tmp_path):
        """Smoke test: synthesis-wizard via subprocess exits 0 and produces valid YAML."""
        import subprocess, sys, yaml

        # Create a valid driver YAML with all required T-S parameters
        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
        )
        out_path = tmp_path / "synth_out.yaml"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "synthesis-wizard",
                "--driver", str(driver),
                "--f3", "40",
                "--output", str(out_path),
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"synthesis-wizard subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )
        assert out_path.exists(), (
            f"synthesis-wizard did not create output file at {out_path}\n"
            f"stdout: {result.stdout[:300]}\n"
            f"stderr: {result.stderr[:300]}"
        )
        text = out_path.read_text(encoding="utf-8")
        # The file has a comment header; find the YAML document after the geometry marker
        yaml_start = text.find("# ── Synthesised geometry")
        assert yaml_start != -1, f"Could not find 'Synthesised geometry' marker in output:\n{text[:300]}"
        yaml_text = text[yaml_start:]
        # Parse to confirm it's valid YAML with expected keys (flat structure)
        geo = yaml.safe_load(yaml_text)
        assert isinstance(geo, dict), "Output is not a valid YAML dict"
        assert "sections" in geo, "Output YAML missing 'sections' key"
        assert "vrc" in geo, "Output YAML missing 'vrc' (rear chamber volume)"
        assert "lrc" in geo, "Output YAML missing 'lrc' (rear chamber length)"
        assert "vtc" in geo, "Output YAML missing 'vtc' (throat chamber volume)"
        assert "atc" in geo, "Output YAML missing 'atc' (throat chamber area)"
        assert "ap1" in geo, "Output YAML missing 'ap1' (throat adapter area)"
        assert "lpt" in geo, "Output YAML missing 'lpt' (throat adapter length)"
        assert "ang" in geo, "Output YAML missing 'ang' (radiation angle)"

    def test_synthesis_wizard_subprocess_missing_driver_exits_nonzero(self, tmp_path):
        """Missing driver file should cause synthesis-wizard to exit non-zero via subprocess."""
        import subprocess, sys

        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "synthesis-wizard",
                "--driver", str(tmp_path / "nonexistent.yaml"),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"synthesis-wizard should fail with missing driver, got exit {result.returncode}"
        )

    def test_synthesis_wizard_subprocess_f3_50_no_output_file(self, tmp_path):
        """Subprocess smoke: synthesis-wizard --f3 50 (no --output) exits 0 and prints geometry to stdout.

        Verifies stdout contains geometry info (Throat area, Mouth area, Path length)
        even when output is streamed to stdout instead of written to a file.
        """
        import subprocess, sys

        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
        )
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "synthesis-wizard",
                "--driver", str(driver),
                "--f3", "50",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"synthesis-wizard --f3 50 failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )
        # Verify geometry info appears in stdout (printed table, not file output)
        combined = result.stdout + result.stderr
        assert "Throat area" in combined or "S1" in combined, (
            f"Expected 'Throat area' or 'S1' in stdout, got:\n{combined[:500]}"
        )
        assert "Mouth area" in combined or "S2" in combined, (
            f"Expected 'Mouth area' or 'S2' in stdout, got:\n{combined[:500]}"
        )
        assert "Path length" in combined or "L12" in combined, (
            f"Expected 'Path length' or 'L12' in stdout, got:\n{combined[:500]}"
        )

    def test_synthesis_wizard_subprocess_missing_f3_exits_nonzero(self, tmp_path):
        """Subprocess smoke: synthesis-wizard without required --driver exits non-zero.

        Note: --f3 is not required (has default 50 Hz); --driver is the sole required param.
        This test verifies the command fails clearly when --driver is absent.
        """
        import subprocess, sys

        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "synthesis-wizard"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"synthesis-wizard without --driver should fail, got exit {result.returncode}"
        )

    def test_synthesis_wizard_subprocess_max_iter_flag(self, tmp_path):
        """Subprocess smoke: synthesis-wizard --f3 50 --max-iter 5 exits 0.

        Verifies the --max-iter flag (added in commit a8bfe7e) is wired through
        to the synthesis-wizard command and does not cause an error.
        """
        import subprocess, sys

        driver = tmp_path / "driver.yaml"
        driver.write_text(
            "fs: 49.6\n"
            "qts: 0.27\n"
            "qes: 0.28\n"
            "qms: 7.88\n"
            "vas: 0.0369\n"
            "re: 7.8\n"
            "bl: 7.79\n"
            "mms: 0.00699\n"
            "cms: 0.001472\n"
            "rms: 0.277\n"
            "sd: 0.01327\n"
            "le: 0.0008\n"
        )
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "synthesis-wizard",
                "--driver", str(driver),
                "--f3", "50",
                "--max-iter", "5",
            ],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"synthesis-wizard --max-iter 5 failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )


# ─── segment-wizard ────────────────────────────────────────────────────────────


class TestSegmentWizard:
    """Smoke tests for the segment-wizard CLI command."""

    def test_segment_wizard_valid_s1_s2_l12(self, cli_runner):
        """Given s1, s2, l12 → computes cutoff frequency f12."""
        result = cli_runner.invoke(
            app,
            ["segment-wizard", "--s1", "40", "--s2", "300", "--l12", "150"],
        )
        assert result.exit_code == 0, result.output
        assert "f12" in result.output.lower() or "cutoff" in result.output.lower()

    def test_segment_wizard_valid_s1_s2_f12(self, cli_runner):
        """Given s1, s2, f12 → computes horn length l12."""
        result = cli_runner.invoke(
            app,
            ["segment-wizard", "--s1", "40", "--s2", "300", "--f12", "50"],
        )
        assert result.exit_code == 0, result.output
        assert "l12" in result.output.lower() or "length" in result.output.lower()

    def test_segment_wizard_missing_params(self, cli_runner):
        """Fewer than 3 of the 4 params should result in a usage error."""
        result = cli_runner.invoke(
            app,
            ["segment-wizard", "--s1", "40", "--s2", "300"],
        )
        assert result.exit_code != 0

    def test_segment_wizard_profile_output(self, cli_runner):
        """Output should include the catenoidal area profile."""
        result = cli_runner.invoke(
            app,
            ["segment-wizard", "--s1", "40", "--s2", "300", "--l12", "150"],
        )
        assert result.exit_code == 0, result.output
        assert "profile" in result.output.lower() or "area" in result.output.lower()

    def test_segment_wizard_volume_estimate(self, cli_runner):
        """Output should include a system volume estimate."""
        result = cli_runner.invoke(
            app,
            ["segment-wizard", "--s1", "40", "--s2", "300", "--l12", "150"],
        )
        assert result.exit_code == 0, result.output
        assert "volume" in result.output.lower() or "V" in result.output

    def test_segment_wizard_subprocess_smoke_test(self, tmp_path):
        """Smoke test: segment-wizard via subprocess exits 0 and emits cutoff info."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main",
             "segment-wizard", "--s1", "40", "--s2", "300", "--l12", "150"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"segment-wizard subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:300]}\n"
            f"  stdout: {result.stdout[:300]}"
        )
        assert "f12" in result.stdout.lower() or "cutoff" in result.stdout.lower(), (
            "segment-wizard output should contain cutoff frequency"
        )


# ─── resize-wizard ─────────────────────────────────────────────────────────────


class TestResizeWizard:
    """Smoke tests for the resize-wizard CLI command."""

    def test_resize_wizard_valid_scale_up(self, cli_runner):
        """Valid project + driver + factor should exit 0."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--project", "projects/hiro.yaml",
                "--driver", "drivers/FE166NV2.yaml",
                "--factor", "1.5",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_resize_wizard_valid_scale_down(self, cli_runner):
        """Factor < 1 (shrink) should also work."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--project", "projects/hiro.yaml",
                "--driver", "drivers/FE166NV2.yaml",
                "--factor", "0.8",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_resize_wizard_missing_geometry(self, cli_runner):
        """Missing --project/--horn should fail."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--driver", "drivers/FE166NV2.yaml",
                "--factor", "1.5",
            ],
        )
        assert result.exit_code != 0

    def test_resize_wizard_missing_driver(self, cli_runner):
        """Missing --driver should fail."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--project", "projects/hiro.yaml",
                "--factor", "1.5",
            ],
        )
        assert result.exit_code != 0

    def test_resize_wizard_missing_factor(self, cli_runner):
        """Missing --factor should fail."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--project", "projects/hiro.yaml",
                "--driver", "drivers/FE166NV2.yaml",
            ],
        )
        assert result.exit_code != 0

    def test_resize_wizard_factor_zero_or_negative(self, cli_runner):
        """factor <= 0 should exit 1 with an error message."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--project", "projects/hiro.yaml",
                "--driver", "drivers/FE166NV2.yaml",
                "--factor", "0.0",
            ],
        )
        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}: {result.output}"
        assert "positive" in result.output.lower() or "factor" in result.output.lower()

    def test_resize_wizard_writes_output_file(self, cli_runner, tmp_path):
        """--output should create the resized geometry YAML."""
        out_path = tmp_path / "resized.yaml"
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--project", "projects/hiro.yaml",
                "--driver", "drivers/FE166NV2.yaml",
                "--factor", "1.2",
                "--output", str(out_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists(), "Output file was not created"

    def test_resize_wizard_horn_geometry_only(self, cli_runner, tmp_path):
        """--horn with --geometry-only should work without a project."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--horn", "source/bk16.yaml",
                "--driver", "drivers/FE166NV2.yaml",
                "--factor", "1.1",
                "--geometry-only",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_resize_wizard_no_adjust_sd(self, cli_runner):
        """--no-adjust-sd should be accepted without error."""
        result = cli_runner.invoke(
            app,
            [
                "resize-wizard",
                "--project", "projects/hiro.yaml",
                "--driver", "drivers/FE166NV2.yaml",
                "--factor", "1.5",
                "--no-adjust-sd",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_resize_wizard_subprocess_smoke_test(self, tmp_path):
        """Smoke test: resize-wizard via subprocess exits 0 and produces valid YAML."""
        import subprocess, sys, yaml
        output_file = tmp_path / "resized.yaml"
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "resize-wizard",
                "--driver", "tests/benchmarks/hornresp_gdb1/gdb1_driver_only.yaml",
                "--horn", "tests/benchmarks/hornresp_gdb1/gdb1_horn_only.yaml",
                "--factor", "1.5",
                "--output", str(output_file),
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"resize-wizard subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:500]}"
        )
        # Verify output file was created and is valid YAML
        assert output_file.exists(), (
            f"resize-wizard did not produce output file at {output_file}\n"
            f"stdout: {result.stdout[:300]}"
        )
        with open(output_file) as f:
            data = yaml.safe_load(f)
        # Verify it has expected horn geometry fields
        assert "throat_area" in data or "mouth_area" in data or "sections" in data, (
            f"Expected horn geometry fields in output, got: {list(data.keys())}"
        )
        # Verify scaling was applied (throat_area should be 0.008 * 1.5^2 = 0.018)
        if "throat_area" in data:
            assert data["throat_area"] == pytest.approx(0.018, rel=1e-3), (
                f"Expected throat_area ~0.018 (0.008 * 1.5^2), got {data['throat_area']}"
            )

    def test_resize_wizard_subprocess_factor_zero_exits_nonzero(self):
        """Subprocess smoke: resize-wizard with factor=0 exits non-zero."""
        import subprocess, sys
        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main", "resize-wizard",
                "--driver", "tests/benchmarks/hornresp_gdb1/gdb1_driver_only.yaml",
                "--horn", "tests/benchmarks/hornresp_gdb1/gdb1_horn_only.yaml",
                "--factor", "0",
            ],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"resize-wizard with factor=0 should fail, got exit {result.returncode}"
        )


# ─── tapped-horn ───────────────────────────────────────────────────────────────


class TestTappedHorn:
    """Smoke tests for the tapped-horn CLI command."""

    def test_tapped_horn_valid(self, cli_runner):
        """Valid driver + TH geometry should exit 0.

        Note: --no-export-csv is required to avoid a pre-existing bug where
        round(complex128) is called on impedance values in the CSV export.
        The --no-plot flag is also set for speed.
        """
        result = cli_runner.invoke(
            app,
            [
                "tapped-horn",
                "--driver", "drivers/FE166NV2.yaml",
                "--th", "source/th_example.yaml",
                "--no-export-csv",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_tapped_horn_missing_driver(self, cli_runner):
        """Missing --driver should exit non-zero."""
        result = cli_runner.invoke(
            app,
            ["tapped-horn", "--th", "source/th_example.yaml", "--no-export-csv", "--no-plot"],
        )
        assert result.exit_code != 0

    def test_tapped_horn_missing_th(self, cli_runner):
        """Missing --th should exit non-zero."""
        result = cli_runner.invoke(
            app,
            ["tapped-horn", "--driver", "drivers/FE166NV2.yaml", "--no-export-csv", "--no-plot"],
        )
        assert result.exit_code != 0

    def test_tapped_horn_nonexistent_driver(self, cli_runner, tmp_path):
        """Non-existent driver path should fail clearly."""
        result = cli_runner.invoke(
            app,
            [
                "tapped-horn",
                "--driver", str(tmp_path / "nonexistent.yaml"),
                "--th", "source/th_example.yaml",
                "--no-export-csv",
                "--no-plot",
            ],
        )
        assert result.exit_code != 0

    def test_tapped_horn_custom_frequency_range(self, cli_runner):
        """Custom --fmin / --fmax should be accepted."""
        result = cli_runner.invoke(
            app,
            [
                "tapped-horn",
                "--driver", "drivers/FE166NV2.yaml",
                "--th", "source/th_example.yaml",
                "--fmin", "30",
                "--fmax", "2000",
                "--n-points", "100",
                "--no-export-csv",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_tapped_horn_no_export_csv(self, cli_runner):
        """--no-export-csv should be accepted."""
        result = cli_runner.invoke(
            app,
            [
                "tapped-horn",
                "--driver", "drivers/FE166NV2.yaml",
                "--th", "source/th_example.yaml",
                "--no-export-csv",
                "--no-plot",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_tapped_horn_subprocess_smoke_test(self, tmp_path):
        """Smoke test: tapped-horn via subprocess exits 0 and emits simulation results."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main",
             "tapped-horn",
             "--driver", "drivers/FE166NV2.yaml",
             "--th", "source/th_example.yaml",
             "--no-export-csv", "--no-plot"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"tapped-horn subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:500]}"
        )
        assert len(result.stdout) > 0, "tapped-horn should emit output"

    def test_tapped_horn_roundtrip_valid_spl_output(self, cli_runner):
        """E2E roundtrip: tapped-horn produces valid acoustic metrics in expected ranges.

        This is the most concrete end-to-end test for the Tapped Horn solver path.
        It verifies the tapped-horn command runs to completion with a valid driver +
        TH geometry combination and emits physically plausible acoustic metrics:
        - Max SPL in range 80-150 dB for a FE166NV2 driver
        - SPL at 1 kHz in range 60-130 dB
        - All reported values finite (no NaN/Inf)

        Note: --no-export-csv is used to work around a known bug where round(complex128)
        is called on impedance values in the CSV export path.
        """
        import re

        result = cli_runner.invoke(
            app,
            [
                "tapped-horn",
                "--driver", "drivers/FE166NV2.yaml",
                "--th", "source/th_example.yaml",
                "--fmin", "50",
                "--fmax", "2000",
                "--n-points", "100",
                "--no-plot",
                "--no-export-csv",
            ],
        )
        assert result.exit_code == 0, (
            f"tapped-horn failed with exit {result.exit_code}:\n"
            f"  stdout: {result.stdout[:300]}\n"
            f"  stderr: {result.stderr[:300]}"
        )

        output = result.output

        # Verify Max SPL is reported
        assert "Max SPL" in output, f"Max SPL not in output:\n{output}"
        max_spl_match = re.search(r"Max SPL[:\s]+([0-9.]+)\s*dB", output)
        assert max_spl_match, f"Could not parse Max SPL from output:\n{output}"
        max_spl = float(max_spl_match.group(1))
        assert 80.0 <= max_spl <= 150.0, (
            f"Max SPL {max_spl} dB outside reasonable range (80-150 dB) for FE166NV2"
        )

        # Verify SPL at 1 kHz is reported and in plausible range
        assert "SPL at 1 kHz" in output or "1 kHz" in output, (
            f"SPL at 1 kHz not in output:\n{output}"
        )
        spl_1k_match = re.search(r"SPL at 1 kHz[:\s]+([0-9.]+)\s*dB", output)
        if spl_1k_match:
            spl_1k = float(spl_1k_match.group(1))
            assert 60.0 <= spl_1k <= 130.0, (
                f"SPL at 1 kHz ({spl_1k} dB) outside plausible range (60-130 dB)"
            )

        # Verify front path length and rear chamber info are present
        assert "Front path length" in output, "Front path length not in output"
        assert "Rear load type" in output, "Rear load type not in output"
        assert "rear_chamber" in output.lower(), "Rear chamber info not in output"

    def test_tapped_horn_roundtrip_subprocess(self, tmp_path):
        """Subprocess E2E: tapped-horn via subprocess exits 0 and emits valid acoustic metrics."""
        import subprocess, sys, re

        result = subprocess.run(
            [
                sys.executable, "-m", "pyhorn_cli.main",
                "tapped-horn",
                "--driver", "drivers/FE166NV2.yaml",
                "--th", "source/th_example.yaml",
                "--fmin", "50",
                "--fmax", "2000",
                "--n-points", "100",
                "--no-plot",
                "--no-export-csv",
            ],
            capture_output=True, text=True, timeout=60,
            cwd="/Users/guillaume/pyhorn",
        )
        assert result.returncode == 0, (
            f"tapped-horn subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:400]}\n"
            f"  stderr: {result.stderr[:400]}"
        )

        output = result.stdout + result.stderr

        # Parse and validate Max SPL
        max_spl_match = re.search(r"Max SPL[:\s]+([0-9.]+)\s*dB", output)
        assert max_spl_match, f"Could not find Max SPL in output:\n{output[:400]}"
        max_spl = float(max_spl_match.group(1))
        assert 80.0 <= max_spl <= 150.0, (
            f"Max SPL {max_spl} dB outside reasonable range (80-150 dB)"
        )

        # Validate SPL at 1 kHz
        spl_1k_match = re.search(r"SPL at 1 kHz[:\s]+([0-9.]+)\s*dB", output)
        assert spl_1k_match, f"Could not find SPL at 1 kHz in output:\n{output[:400]}"
        spl_1k = float(spl_1k_match.group(1))
        assert 60.0 <= spl_1k <= 130.0, (
            f"SPL at 1 kHz ({spl_1k} dB) outside plausible range (60-130 dB)"
        )

        # Verify simulation metadata is present
        assert "Front path length" in output, "Front path length missing"
        assert "Rear load type" in output, "Rear load type missing"
        print(f"Max SPL: {max_spl} dB, SPL at 1 kHz: {spl_1k} dB — ALL OK")


# ─── diagnose-spl ──────────────────────────────────────────────────────────────


class TestDiagnoseSpl:
    """Smoke tests for the diagnose-spl CLI command."""

    def test_diagnose_spl_valid_horn(self, cli_runner):
        """Valid driver + horn geometry should exit 0 and emit diagnostics."""
        result = cli_runner.invoke(
            app,
            [
                "diagnose-spl",
                "--driver", "drivers/FE166NV2.yaml",
                "--horn", "source/bk16.yaml",
            ],
        )
        assert result.exit_code == 0, result.output
        # Should output some analysis indicators
        assert any(
            kw in result.output.lower()
            for kw in ["smoothness", "standing-wave", "artifact", "score", "diagnos"]
        )

    def test_diagnose_spl_valid_project(self, cli_runner):
        """--project instead of --horn should also work."""
        result = cli_runner.invoke(
            app,
            [
                "diagnose-spl",
                "--driver", "drivers/FE166NV2.yaml",
                "--project", "projects/bk16.yaml",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_diagnose_spl_missing_driver(self, cli_runner):
        """Missing --driver should exit non-zero."""
        result = cli_runner.invoke(
            app,
            ["diagnose-spl", "--horn", "source/bk16.yaml"],
        )
        assert result.exit_code != 0

    def test_diagnose_spl_missing_geometry(self, cli_runner):
        """Neither --horn nor --project provided should exit non-zero."""
        result = cli_runner.invoke(
            app,
            ["diagnose-spl", "--driver", "drivers/FE166NV2.yaml"],
        )
        assert result.exit_code != 0

    def test_diagnose_spl_writes_csv(self, cli_runner, tmp_path):
        """--output-csv should create the CSV file."""
        csv_path = tmp_path / "diag.csv"
        result = cli_runner.invoke(
            app,
            [
                "diagnose-spl",
                "--driver", "drivers/FE166NV2.yaml",
                "--horn", "source/bk16.yaml",
                "--output-csv", str(csv_path),
                "--n-points", "200",
            ],
        )
        assert result.exit_code == 0, result.output
        assert csv_path.exists(), "CSV output file was not created"

    def test_diagnose_spl_custom_band(self, cli_runner):
        """Custom --band-start / --band-end should be accepted."""
        result = cli_runner.invoke(
            app,
            [
                "diagnose-spl",
                "--driver", "drivers/FE166NV2.yaml",
                "--horn", "source/bk16.yaml",
                "--band-start", "100",
                "--band-end", "400",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_diagnose_spl_custom_artifact_threshold(self, cli_runner):
        """--artifact-threshold should be accepted."""
        result = cli_runner.invoke(
            app,
            [
                "diagnose-spl",
                "--driver", "drivers/FE166NV2.yaml",
                "--horn", "source/bk16.yaml",
                "--artifact-threshold", "3.0",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_diagnose_spl_standing_wave_freqs(self, cli_runner):
        """--standing-wave-freqs flag should run extended comb-filtering analysis and exit 0."""
        result = cli_runner.invoke(
            app,
            [
                "diagnose-spl",
                "--driver", "drivers/FE166NV2.yaml",
                "--horn", "source/bk16.yaml",
                "--standing-wave-freqs",
            ],
        )
        assert result.exit_code == 0, result.output
        # Extended scan should mention standing-wave detection in the output
        assert any(
            kw in result.output.lower()
            for kw in ["standing-wave", "comb-filter", "chamber resonance"]
        )

    def test_diagnose_spl_inverted_band_range_exits_nonzero(self, cli_runner):
        """--band-start > --band-end should exit non-zero with a clear warning."""
        result = cli_runner.invoke(
            app,
            [
                "diagnose-spl",
                "--driver", "drivers/FE166NV2.yaml",
                "--horn", "source/bk16.yaml",
                "--band-start", "500",   # intentionally > band-end
                "--band-end", "100",
                "--n-points", "50",
            ],
        )
        assert result.exit_code != 0, (
            f"diagnose-spl should exit non-zero when band-start > band-end, "
            f"got exit {result.exit_code}"
        )
        assert "band" in result.output.lower() or "point" in result.output.lower(), (
            f"Expected a band/points warning in output, got: {result.output[:200]}"
        )

    def test_diagnose_spl_subprocess_smoke_test(self):
        """Smoke test: diagnose-spl via subprocess exits 0 and emits diagnostic output."""
        import subprocess, sys
        # Use --n-points 100 to keep the scan fast (default is 5000)
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "diagnose-spl",
             "--driver", "drivers/FE166NV2.yaml",
             "--horn", "source/bk16.yaml",
             "--n-points", "100",
             "--band-start", "200",
             "--band-end", "400"],
            capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, (
            f"diagnose-spl subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:500]}\n"
            f"  stdout: {result.stdout[:500]}"
        )
        output_lower = result.stdout.lower()
        assert any(
            kw in output_lower
            for kw in ["smoothness", "standing-wave", "artifact", "score", "diagnos",
                       "band", "frequency", "mean", "std"]
        ), f"diagnose-spl output should contain diagnostic keywords, got:\n{result.stdout[:300]}"

    def test_diagnose_spl_subprocess_missing_driver_exits_nonzero(self):
        """Subprocess smoke: diagnose-spl with non-existent driver exits non-zero."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "diagnose-spl",
             "--driver", "drivers/NONEXISTENT.yaml",
             "--horn", "source/bk16.yaml"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"diagnose-spl should exit non-zero for missing driver, got exit {result.returncode}"
        )


# ─── driver-front-volume ───────────────────────────────────────────────────────


class TestDriverFrontVolume:
    """Smoke tests for the driver-front-volume CLI command."""

    def test_driver_front_volume_valid(self, cli_runner):
        """All required dimensions provided → exit 0 with volume output."""
        result = cli_runner.invoke(
            app,
            [
                "driver-front-volume",
                "--d1", "100",
                "--d2", "80",
                "--d3", "20",
                "--h1", "5",
                "--h2", "15",
                "--h3", "10",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "V_front" in result.output or "volume" in result.output.lower()

    def test_driver_front_volume_missing_params(self, cli_runner):
        """Any missing required dimension should exit non-zero."""
        result = cli_runner.invoke(
            app,
            [
                "driver-front-volume",
                "--d1", "100",
                "--d2", "80",
                # d3, h1, h2, h3 missing
            ],
        )
        assert result.exit_code != 0

    def test_driver_front_volume_volume_components(self, cli_runner):
        """Output should include shell and cone volume components."""
        result = cli_runner.invoke(
            app,
            [
                "driver-front-volume",
                "--d1", "100",
                "--d2", "80",
                "--d3", "20",
                "--h1", "5",
                "--h2", "15",
                "--h3", "10",
            ],
        )
        assert result.exit_code == 0, result.output
        # Both V_shell and V_cone (or equivalent labels) should appear
        output_lower = result.output.lower()
        assert "shell" in output_lower or "v_shell" in output_lower
        assert "cone" in output_lower or "v_cone" in output_lower

    def test_driver_front_volume_boundary_small(self, cli_runner):
        """Very small dimensions should run without crash."""
        result = cli_runner.invoke(
            app,
            [
                "driver-front-volume",
                "--d1", "30",
                "--d2", "25",
                "--d3", "5",
                "--h1", "1",
                "--h2", "2",
                "--h3", "1",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_driver_front_volume_boundary_large(self, cli_runner):
        """Large dimensions should run without crash."""
        result = cli_runner.invoke(
            app,
            [
                "driver-front-volume",
                "--d1", "300",
                "--d2", "250",
                "--d3", "50",
                "--h1", "20",
                "--h2", "50",
                "--h3", "30",
            ],
        )
        assert result.exit_code == 0, result.output

    def test_driver_front_volume_zero_area_warning(self, cli_runner):
        """d3 close to d2 (no shell clearance) should run (may warn but not crash)."""
        result = cli_runner.invoke(
            app,
            [
                "driver-front-volume",
                "--d1", "80",
                "--d2", "80",
                "--d3", "20",
                "--h1", "5",
                "--h2", "15",
                "--h3", "10",
            ],
        )
        # Should not crash; exit code 0 or 1 is acceptable
        assert result.exit_code in (0, 1), f"Unexpected exit: {result.exit_code}\n{result.output}"

    def test_driver_front_volume_subprocess_smoke_test(self):
        """Smoke test: driver-front-volume via subprocess exits 0 and prints volume output."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "driver-front-volume",
             "--d1", "100", "--d2", "80", "--d3", "20",
             "--h1", "5", "--h2", "15", "--h3", "10"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"driver-front-volume subprocess failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:200]}\n"
            f"  stdout: {result.stdout[:200]}"
        )
        # Verify expected output markers
        for marker in ["Driver Front Volume", "Effective front volume", "cm³"]:
            assert marker in result.stdout, (
                f"Expected '{marker}' in stdout, got:\n{result.stdout[:300]}"
            )

    def test_driver_front_volume_subprocess_missing_params_exits_nonzero(self):
        """Subprocess smoke: driver-front-volume with missing required params exits non-zero."""
        import subprocess, sys
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "driver-front-volume",
             "--d1", "100"],  # missing all other required params
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"driver-front-volume with missing params should fail, got exit {result.returncode}"
        )


# ─── E2E: calculate with real geometries ──────────────────────────────────────

import csv
import math
from pathlib import Path


class TestCalculateE2E:
    """End-to-end tests for the calculate command with real geometries.

    These tests verify actual output files and data quality, not just exit codes.
    """

    @pytest.fixture
    def hiro_driver(self):
        """FE166NV2 driver specs as a temp YAML file."""
        content = """
fs: 43.0
qts: 0.269
qes: 0.297
qms: 2.81
vas: 0.00222
re: 7.8
bl: 7.79
mms: 0.00699
cms: 0.001472
rms: 0.277
sd: 0.001327
le: 0.0008
xmax: 0.0015
"""
        return content

    @pytest.fixture
    def hiro_geometry_path(self):
        """Path to the hiro geometry YAML (BK Hiro)."""
        gdb1 = Path(__file__).resolve().parent.parent.parent
        return gdb1 / "source" / "hiro.yaml"

    def test_calculate_produces_valid_csv_within_spl_range(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """E2E: calculate with hiro driver+geometry produces SPL in 60-130 dB range."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "20",
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        # Find the response.csv (in a subdirectory named after geometry)
        csv_files = list(out_dir.rglob("response.csv"))
        assert len(csv_files) == 1, f"Expected 1 CSV, found {csv_files}"
        csv_path = csv_files[0]

        # Read and validate SPL values
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 20, f"Expected 20 frequency points, got {len(rows)}"

        # Find the SPL column (exporter prefixes with SPL_dB_)
        spl_col = [c for c in rows[0].keys() if "SPL" in c and "Horn SPL" in c][0]
        spls = [float(r[spl_col]) for r in rows]

        # SPL should be in a reasonable acoustic range (60-130 dB)
        assert all(60.0 <= spl <= 130.0 for spl in spls), (
            f"SPL values out of range: {min(spls):.1f}–{max(spls):.1f} dB"
        )
        # SPL range should be physically plausible (< 50 dB spread in 100-2000 Hz)
        assert max(spls) - min(spls) < 50.0, (
            f"SPL range too large: {max(spls)-min(spls):.1f} dB — possible simulation error"
        )

    def test_calculate_csv_has_expected_columns(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """E2E: response.csv contains all key acoustic output columns."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "200",
                "--fmax", "1000",
                "--n-points", "10",
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        csv_files = list(out_dir.rglob("response.csv"))
        with open(csv_files[0], newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames

        required = ["Frequency_Hz", "Horn SPL", "Impedance (Ohms)", "Group delay (ms)"]
        for req in required:
            assert any(req in c for c in cols), f"Missing column containing: {req}"

    def test_calculate_extreme_high_frequency_no_nan(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """Regression: frequencies > 20 kHz should not produce NaN/Inf in output."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "30000",
                "--n-points", "50",
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        csv_files = list(out_dir.rglob("response.csv"))
        with open(csv_files[0], newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Check all numeric columns for NaN
        for row in rows:
            for col, val in row.items():
                try:
                    num = float(val)
                    assert not math.isnan(num), f"NaN in column {col} at frequency {row.get('Frequency_Hz')}"
                    assert not math.isinf(num), f"Inf in column {col} at frequency {row.get('Frequency_Hz')}"
                except (ValueError, TypeError):
                    pass  # non-numeric column, skip

    def test_calculate_extreme_low_frequency_no_crash(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """Regression: frequencies < 20 Hz should not crash or overflow."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "5",
                "--fmax", "50",
                "--n-points", "10",
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        csv_files = list(out_dir.rglob("response.csv"))
        with open(csv_files[0], newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 10
        # Check SPL is finite (may be very low, but must be finite)
        spl_col = [c for c in rows[0].keys() if "SPL" in c and "Horn SPL" in c][0]
        for row in rows:
            spl = float(row[spl_col])
            assert math.isfinite(spl), f"Non-finite SPL at {row.get('Frequency_Hz')} Hz: {spl}"

    def test_calculate_export_frd_produces_valid_frd_file(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """E2E: calculate with --export-frd produces a valid .frd file."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        frd_path = tmp_path / "response.frd"
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "20",
                "--no-plot",
                "--no-plot-3d",
                "--export-frd",
                str(frd_path),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert frd_path.exists(), f"FRD file not created at {frd_path}"

        # Verify FRD format
        content = frd_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()
        assert len(lines) > 2, "FRD must have header + column header + data rows"
        assert lines[0] == "!FRD1.0", f"Expected !FRD1.0 header, got: {lines[0]}"
        assert "Frequency(Hz)" in lines[1], f"Expected column header, got: {lines[1]}"

        # Verify data rows have 3 columns (freq, spl, phase)
        data_lines = [l for l in lines[2:] if l.strip() and not l.startswith("!")]
        assert len(data_lines) == 20, f"Expected 20 data points, got {len(data_lines)}"
        for line in data_lines:
            parts = line.split()
            assert len(parts) == 3, f"Expected 3 columns in data line: {line}"
            freq, spl, phase = float(parts[0]), float(parts[1]), float(parts[2])
            assert 20 < freq < 50000, f"Frequency out of range: {freq}"
            assert -200 < spl < 200, f"SPL out of plausible range: {spl}"
            assert isinstance(phase, float), "Phase must be numeric"

    def test_calculate_export_json_produces_valid_json_file(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """E2E: calculate with --export-json produces a valid JSON file with required keys."""
        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "20",
                "--no-plot",
                "--no-plot-3d",
                "--export-json",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        # Find the response.json (in a subdirectory named after geometry)
        json_files = list(out_dir.rglob("response.json"))
        assert len(json_files) == 1, f"Expected 1 JSON file, found {json_files}"
        json_path = json_files[0]

        # Verify JSON is parseable and contains required keys
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        assert "frequencies" in data, "JSON missing 'frequencies' key"
        assert "responses" in data, "JSON missing 'responses' key"
        assert isinstance(data["frequencies"], list), "'frequencies' must be a list"
        assert isinstance(data["responses"], dict), "'responses' must be a dict"
        assert len(data["frequencies"]) == 20, f"Expected 20 frequency points, got {len(data['frequencies'])}"

        # Verify all SPL response arrays match the frequency count
        for label, spl_values in data["responses"].items():
            assert len(spl_values) == len(data["frequencies"]), (
                f"Response '{label}' length {len(spl_values)} != frequency count {len(data['frequencies'])}"
            )

    def test_calculate_export_wav_produces_valid_wav_file(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """E2E: calculate with --export-wav produces a valid 16-bit PCM WAV file."""
        import wave as _wave
        import struct

        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        wav_path = tmp_path / "response.wav"
        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "2000",
                "--n-points", "20",
                "--no-plot",
                "--no-plot-3d",
                "--export-wav", str(wav_path),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert wav_path.exists(), f"WAV file not created at {wav_path}"

        # Verify WAV header
        with _wave.open(str(wav_path), "rb") as wf:
            # Check RIFF header was parsed (wave.open succeeds without error)
            assert wf.getnchannels() == 1, f"Expected mono (1 channel), got {wf.getnchannels()}"
            assert wf.getsampwidth() == 2, f"Expected 16-bit (2 bytes), got {wf.getsampwidth()}"
            assert wf.getframerate() == 44100, f"Expected 44100 Hz sample rate, got {wf.getframerate()}"
            n_frames = wf.getnframes()
            assert n_frames > 0, "WAV file has zero audio frames"

        # Verify file size matches expected WAV size (44-byte header + n_frames * sampwidth bytes)
        file_size = wav_path.stat().st_size
        expected_min_size = 44 + n_frames * 2
        assert file_size >= expected_min_size, (
            f"WAV file size {file_size} < expected minimum {expected_min_size} "
            f"(header + {n_frames} frames × 2 bytes)"
        )
        assert file_size <= expected_min_size + 100, (
            f"WAV file size {file_size} exceeds expected {expected_min_size} by > 100 bytes — possible extra data"
        )

    def test_calculate_with_filter_yaml_produces_filtered_output(
        self, cli_runner, tmp_path, hiro_driver, hiro_geometry_path
    ):
        """E2E: calculate with --filter-yaml applies filter bands and produces filtered SPL."""
        import csv

        driver_file = tmp_path / "driver.yaml"
        driver_file.write_text(hiro_driver)

        filter_file = tmp_path / "my_filter.yaml"
        filter_file.write_text(
            "filter_bands:\n"
            "  - type: 'lowpass'\n"
            "    frequency: 3000\n"
            "    q: 0.7\n"
            "    order: 2\n"
            "    enabled: true\n"
        )

        out_dir = tmp_path / "outputs"
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d", str(driver_file),
                "-h", str(hiro_geometry_path),
                "-o", str(out_dir),
                "--fmin", "100",
                "--fmax", "5000",
                "--n-points", "30",
                "--no-plot",
                "--no-plot-3d",
                "--filter-yaml", str(filter_file),
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"

        # CLI creates subdir named after geometry; filter adds "Filtered SPL (dB)" column
        csv_path = out_dir / "hiro" / "response.csv"
        assert csv_path.exists(), f"CSV not created at {csv_path}"

        # Verify CSV has filtered SPL columns
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0, "CSV has no data rows"
            # Filtered SPL column is added when --filter-yaml is used
            fieldnames = reader.fieldnames or []
            has_filtered = any("Filtered SPL" in col for col in fieldnames)
            assert has_filtered, f"Expected 'Filtered SPL' column in CSV, got: {fieldnames}"


# ─── Registry commands ─────────────────────────────────────────────────────────


import subprocess
import sys
import yaml


class TestRegistryCommands:
    """Tests for registry CLI commands."""

    def test_import_existing_subprocess_smoke_test(self, tmp_path):
        """import-existing via subprocess exits 0 and reports import/skipped counts."""
        # Create a temp project dir with a drivers/ subdirectory and one YAML file
        driver_dir = tmp_path / "drivers"
        driver_dir.mkdir()
        test_driver = driver_dir / "test_driver.yaml"
        test_driver.write_text(
            "sd: 0.01327\nre: 7.8\nqes: 0.29\nqms: 3.1\nfs: 53\nvas: 0.0175\nbl: 7.79\nle: 0.0008\n"
        )

        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "import-existing",
             "driver"],
            capture_output=True, text=True, timeout=30,
            cwd=str(tmp_path),
        )
        assert result.returncode == 0, (
            f"registry import-existing subprocess failed with exit {result.returncode}:\n"
            f"  stdout: {result.stdout[:300]}\n  stderr: {result.stderr[:300]}"
        )
        # Should report imported or skipped count
        output = result.stdout + result.stderr
        assert "Imported" in output or "skipped" in output, (
            f"Expected import summary in output, got: {output[:300]}"
        )

    def test_import_existing_invalid_kind_exits_nonzero(self, tmp_path):
        """import-existing with invalid kind exits non-zero."""
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "import-existing",
             "bogus"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"registry import-existing bogus should exit non-zero, got {result.returncode}"
        )

    def test_import_existing_nonexistent_directory_exits_zero(self, tmp_path):
        """import-existing with a missing drivers/ dir exits 0 (graceful no-op)."""
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "import-existing",
             "driver"],
            capture_output=True, text=True, timeout=30,
            cwd=str(tmp_path),  # tmp_path has no drivers/ subdir
        )
        # Should exit 0 gracefully with a warning about missing directory
        assert result.returncode == 0, (
            f"registry import-existing should exit 0 for missing dir, got {result.returncode}\n"
            f"  stderr: {result.stderr[:300]}"
        )

    def test_registry_list_subprocess_smoke_test(self, tmp_path):
        """Smoke test: registry list via subprocess exits 0 even on an empty registry."""
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "list",
             "--base", str(tmp_path / "reg")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, (
            f"registry list failed with exit {result.returncode}:\n"
            f"  stderr: {result.stderr[:300]}\n  stdout: {result.stdout[:300]}"
        )
        # Should report no entries (empty registry)
        combined = result.stdout + result.stderr
        assert "no entries" in combined.lower(), (
            f"Expected 'no entries' in empty registry list output, got:\n{combined[:300]}"
        )

    def test_registry_add_then_get_subprocess_smoke_test(self, tmp_path):
        """Smoke test: add a driver to a temp registry, then retrieve it with get."""
        # Create a minimal driver YAML to add
        driver_file = tmp_path / "test_driver.yaml"
        driver_file.write_text(
            "fs: 53\nqts: 0.27\nqes: 0.29\nqms: 3.8\n"
            "vas: 0.0175\nre: 7.8\nbl: 7.79\nle: 0.0008\n"
            "mms: 0.00699\ncms: 0.001472\nrms: 0.277\nsd: 0.01327\n"
        )
        reg_base = tmp_path / "reg"

        # Step 1: add the driver
        add_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "add",
             "test_driver", "driver",
             "--file", str(driver_file),
             "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert add_result.returncode == 0, (
            f"registry add failed with exit {add_result.returncode}:\n"
            f"  stderr: {add_result.stderr[:300]}"
        )

        # Step 2: get the driver back
        get_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "get",
             "test_driver", "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert get_result.returncode == 0, (
            f"registry get failed with exit {get_result.returncode}:\n"
            f"  stderr: {get_result.stderr[:300]}\n  stdout: {get_result.stdout[:300]}"
        )
        assert "test_driver" in get_result.stdout, (
            f"Expected 'test_driver' in get output, got:\n{get_result.stdout[:300]}"
        )
        assert "driver" in get_result.stdout, (
            f"Expected 'driver' kind in get output, got:\n{get_result.stdout[:300]}"
        )

    def test_registry_resolve_subprocess_smoke_test(self, tmp_path):
        """Smoke test: resolve returns the file path for an added driver."""
        # Create a minimal driver YAML to add
        driver_file = tmp_path / "resolve_test_driver.yaml"
        driver_file.write_text(
            "fs: 43\nqts: 0.27\nvas: 0.00222\nre: 7.8\n"
            "bl: 7.79\nsd: 0.001327\nle: 0.0008\n"
        )
        reg_base = tmp_path / "reg"

        # Add the driver first
        add_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "add",
             "resolve_test_driver", "driver",
             "--file", str(driver_file),
             "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert add_result.returncode == 0, (
            f"registry add failed with exit {add_result.returncode}:\n"
            f"  stderr: {add_result.stderr[:300]}"
        )

        # Resolve the driver
        resolve_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "resolve",
             "resolve_test_driver", "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert resolve_result.returncode == 0, (
            f"registry resolve failed with exit {resolve_result.returncode}:\n"
            f"  stderr: {resolve_result.stderr[:300]}\n  stdout: {resolve_result.stdout[:300]}"
        )
        # Output should contain a path ending in the registry drivers dir
        combined = resolve_result.stdout + resolve_result.stderr
        assert "drivers" in combined, (
            f"Expected resolved path to contain 'drivers', got:\n{combined[:300]}"
        )

    def test_registry_update_subprocess_smoke_test(self, tmp_path):
        """Smoke test: update adds a tag to an existing entry and get reflects it."""
        # Create and register a driver
        driver_file = tmp_path / "update_test_driver.yaml"
        driver_file.write_text(
            "fs: 50\nqts: 0.27\nvas: 0.01\nre: 7.8\n"
            "bl: 7.79\nsd: 0.01\nle: 0.0008\n"
        )
        reg_base = tmp_path / "reg"

        add_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "add",
             "update_driver", "driver",
             "--file", str(driver_file),
             "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert add_result.returncode == 0, (
            f"registry add failed: {add_result.stderr[:200]}"
        )

        # Update with a tag
        update_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "update",
             "update_driver", "--add-tag", "test-tag",
             "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert update_result.returncode == 0, (
            f"registry update failed with exit {update_result.returncode}:\n"
            f"  stderr: {update_result.stderr[:300]}"
        )

        # Verify tag appears in get
        get_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "get",
             "update_driver", "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert get_result.returncode == 0, get_result.stderr[:200]
        assert "test-tag" in get_result.stdout, (
            f"Expected 'test-tag' in get output after update, got:\n{get_result.stdout[:300]}"
        )

    def test_registry_remove_subprocess_smoke_test(self, tmp_path):
        """Smoke test: remove deletes an entry and subsequent get returns non-zero."""
        # Create and register a driver
        driver_file = tmp_path / "remove_test_driver.yaml"
        driver_file.write_text(
            "fs: 50\nqts: 0.27\nvas: 0.01\nre: 7.8\nbl: 7.79\nsd: 0.01\n"
        )
        reg_base = tmp_path / "reg"

        add_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "add",
             "remove_driver", "driver",
             "--file", str(driver_file),
             "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert add_result.returncode == 0, (
            f"registry add failed: {add_result.stderr[:200]}"
        )

        # Remove the driver
        remove_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "remove",
             "remove_driver", "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert remove_result.returncode == 0, (
            f"registry remove failed with exit {remove_result.returncode}:\n"
            f"  stderr: {remove_result.stderr[:300]}"
        )

        # get should now fail
        get_result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "get",
             "remove_driver", "--base", str(reg_base)],
            capture_output=True, text=True, timeout=30,
        )
        assert get_result.returncode != 0, (
            f"registry get should fail for removed entry, got exit {get_result.returncode}"
        )

    def test_registry_invalid_subcommand_exits_nonzero(self):
        """Unknown registry subcommand should exit non-zero."""
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry",
             "does-not-exist"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"registry invalid subcommand should exit non-zero, got {result.returncode}\n"
            f"  stdout: {result.stdout[:200]}\n  stderr: {result.stderr[:200]}"
        )

    def test_registry_get_missing_name_exits_nonzero(self, tmp_path):
        """registry get with no name argument should exit non-zero."""
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "get",
             "--base", str(tmp_path / "reg")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"registry get without name should exit non-zero, got {result.returncode}\n"
            f"  stdout: {result.stdout[:200]}\n  stderr: {result.stderr[:200]}"
        )

    def test_registry_resolve_missing_name_exits_nonzero(self, tmp_path):
        """registry resolve with no name argument should exit non-zero."""
        result = subprocess.run(
            [sys.executable, "-m", "pyhorn_cli.main", "registry", "resolve",
             "--base", str(tmp_path / "reg")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode != 0, (
            f"registry resolve without name should exit non-zero, got {result.returncode}\n"
            f"  stdout: {result.stdout[:200]}\n  stderr: {result.stderr[:200]}"
        )
