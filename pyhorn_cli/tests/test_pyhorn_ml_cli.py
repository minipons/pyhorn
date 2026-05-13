import pytest
from typer.testing import CliRunner

# pyhorn_ml is PAUSED — skip all tests if joblib is not available
pytest.importorskip("joblib", reason="pyhorn_ml is PAUSED — joblib not installed")

from pyhorn_ml.cli.commands import app


def test_pyhorn_ml_help_lists_commands():
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "optimize" in result.output
    assert "train-surrogate" in result.output
    assert "inverse" in result.output
    assert "compare" in result.output


def test_compare_command_delegates_to_run_compare(monkeypatch):
    runner = CliRunner()
    captured = {}

    def fake_run_compare(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("pyhorn_ml.cli.compare.run_compare", fake_run_compare)

    result = runner.invoke(
        app,
        [
            "compare",
            "--designs",
            "a.yaml,b.yaml",
            "--driver",
            "drivers/FE166NV2.yaml",
            "--output",
            "out.png",
            "--fmin",
            "80",
            "--fmax",
            "500",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "designs": "a.yaml,b.yaml",
        "driver": "drivers/FE166NV2.yaml",
        "output": "out.png",
        "fmin": 80.0,
        "fmax": 500.0,
    }


def test_train_surrogate_command_delegates_to_run_train(monkeypatch):
    runner = CliRunner()
    captured = {}

    def fake_run_train(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("pyhorn_ml.cli.train.run_train", fake_run_train)

    result = runner.invoke(
        app,
        [
            "train-surrogate",
            "--data-dir",
            "outputs/ml/data/.dataset",
            "--model-type",
            "mlp",
            "--save-as",
            "model.pkl",
        ],
    )

    assert result.exit_code == 0
    assert captured == {
        "data_dir": "outputs/ml/data/.dataset",
        "model_type": "mlp",
        "save_as": "model.pkl",
    }
