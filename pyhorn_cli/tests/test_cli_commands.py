"""Focused CLI regression tests for supported commands."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from pyhorn_cli.main import app
from pyhorn_core.config.models import DriverSpecs
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_core.solver.optimizer import OptimizationResult


@pytest.fixture
def cli_runner():
    return CliRunner()


@pytest.fixture
def driver_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "driver.yaml"
    path.write_text(
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
        "voltage: 2.83\n"
        "le: 0.0008\n"
        "xmax: 0.0015\n"
    )
    return path


@pytest.fixture
def horn_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "horn.yaml"
    path.write_text(
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
    return path


@pytest.fixture
def two_horn_yamls(tmp_path: Path) -> tuple[Path, Path]:
    horn_a = tmp_path / "horn_a.yaml"
    horn_b = tmp_path / "horn_b.yaml"
    horn_a.write_text(
        "enclosure_type: BLH\n"
        "conical_segments:\n"
        "  - [0.006, 0.010, 0.20]\n"
        "  - [0.010, 0.030, 0.25]\n"
    )
    horn_b.write_text(
        "enclosure_type: BLH\n"
        "conical_segments:\n"
        "  - [0.005, 0.009, 0.18]\n"
        "  - [0.009, 0.025, 0.28]\n"
    )
    return horn_a, horn_b


class TestAppStructure:
    def test_help_lists_supported_commands(self, cli_runner: CliRunner):
        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "calculate" in result.output
        assert "compare" in result.output
        assert "derive-ts" in result.output
        assert "hornresp" in result.output
        assert "auto-segment" in result.output
        assert "optimize" in result.output


class TestOptimize:
    def test_optimize_passes_supported_config(
        self, cli_runner: CliRunner, driver_yaml: Path, monkeypatch
    ):
        captured = {}

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

        def fake_optimize(driver_arg, config_arg, **kwargs):
            captured["min_expansion_ratio"] = config_arg.min_expansion_ratio
            captured["throat_area_penalty_weight"] = (
                config_arg.throat_area_penalty_weight
            )
            return [fake_result]

        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands.parse_driver_specs", lambda _: driver
        )
        monkeypatch.setattr(
            "pyhorn_cli.cli.optimize_commands.run_optimize", fake_optimize
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
                str(driver_yaml),
                "--min-expansion-ratio",
                "6.5",
                "--throat-penalty-weight",
                "1.25",
            ],
        )

        assert result.exit_code == 0, result.output
        assert captured["min_expansion_ratio"] == pytest.approx(6.5)
        assert captured["throat_area_penalty_weight"] == pytest.approx(1.25)
        assert "optimized_1_conical.yaml" in result.output


class TestCalculate:
    def test_calculate_requires_project_or_horn(
        self, cli_runner: CliRunner, driver_yaml: Path, tmp_path: Path
    ):
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                str(driver_yaml),
                "-o",
                str(tmp_path / "outputs"),
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code != 0
        assert "Specify either --project or --horn" in result.output

    def test_calculate_missing_horn_file(
        self, cli_runner: CliRunner, driver_yaml: Path, tmp_path: Path
    ):
        result = cli_runner.invoke(
            app,
            [
                "calculate",
                "-d",
                str(driver_yaml),
                "-h",
                str(tmp_path / "missing.yaml"),
                "-o",
                str(tmp_path / "outputs"),
                "--no-plot",
                "--no-plot-3d",
            ],
        )
        assert result.exit_code != 0

    def test_calculate_subprocess_smoke(
        self, driver_yaml: Path, horn_yaml: Path, tmp_path: Path
    ):
        out_dir = tmp_path / "calc_out"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pyhorn_cli.main",
                "calculate",
                "-d",
                str(driver_yaml),
                "-h",
                str(horn_yaml),
                "-o",
                str(out_dir),
                "--fmin",
                "100",
                "--fmax",
                "2000",
                "--n-points",
                "50",
                "--no-plot",
                "--no-plot-3d",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr

        csv_path = out_dir / "horn" / "response.csv"
        assert csv_path.exists()
        with csv_path.open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) >= 10
        spl_keys = [key for key in rows[0] if "SPL" in key and "Horn" in key]
        assert spl_keys


class TestCompare:
    def test_compare_runs(
        self,
        cli_runner: CliRunner,
        driver_yaml: Path,
        two_horn_yamls: tuple[Path, Path],
        tmp_path: Path,
    ):
        horn_a, horn_b = two_horn_yamls
        out_dir = tmp_path / "compare_out"
        result = cli_runner.invoke(
            app,
            [
                "compare",
                str(horn_a),
                str(horn_b),
                "-d",
                str(driver_yaml),
                "-o",
                str(out_dir),
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "spl_compare.png").exists()

    def test_compare_reports_missing_horns_without_traceback(
        self, cli_runner: CliRunner, driver_yaml: Path, tmp_path: Path
    ):
        result = cli_runner.invoke(
            app,
            [
                "compare",
                str(tmp_path / "missing_a.yaml"),
                str(tmp_path / "missing_b.yaml"),
                "-d",
                str(driver_yaml),
            ],
        )
        assert "Traceback" not in result.output
