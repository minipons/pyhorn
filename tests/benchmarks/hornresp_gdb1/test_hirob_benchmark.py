import csv
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyhorn_core.config.parser import parse_horn_project, parse_driver_specs
from pyhorn_core.pyhorn_physics.orchestrators import horn_response
from tests.benchmarks.hornresp_gdb1.compare_hirob import (
    BENCHMARK_CROSSOVER_HZ,
    build_hirob_benchmark_curve,
    build_hirob_benchmark_driver,
)

REPO = Path(__file__).resolve().parents[3]
HR_CSV = REPO / "tests/benchmarks/hornresp_gdb1/hornresp_spl_hirob.csv"
PROJECT = REPO / "projects/hirob.yaml"
DRIVER = REPO / "drivers/FE166NV2.yaml"


def _load_hornresp_csv() -> tuple[np.ndarray, np.ndarray]:
    freqs, spls = [], []
    with open(HR_CSV) as f:
        for row in csv.DictReader(f):
            freqs.append(float(row["Freq (hertz)"]))
            spls.append(float(row["SPL (dB)"]))
    return np.array(freqs), np.array(spls)


def test_hirob_benchmark_driver_disables_productized_layers():
    driver = parse_driver_specs(DRIVER)
    benchmark_driver = build_hirob_benchmark_driver(driver)

    assert benchmark_driver.spl_response is None
    assert benchmark_driver.lossy_le is False
    np.testing.assert_allclose(
        benchmark_driver.get_sensitivity_db(np.array([200.0, 1000.0, 5000.0])),
        driver.get_sensitivity_db(np.array([200.0, 1000.0, 5000.0])),
    )


def test_hirob_benchmark_composite_tracks_reference_within_bounds():
    hr_freqs, hr_spls = _load_hornresp_csv()
    _, horn = parse_horn_project(PROJECT)
    driver = build_hirob_benchmark_driver(parse_driver_specs(DRIVER))

    py_freqs = np.logspace(
        np.log10(max(hr_freqs.min(), 10.0)),
        np.log10(min(hr_freqs.max(), 20000.0)),
        1500,
    )
    result = horn_response(py_freqs, driver, horn, compute_distortion=False)
    _, py_pb, benchmark_curve = build_hirob_benchmark_curve(
        hr_freqs,
        py_freqs,
        result,
        crossover_hz=BENCHMARK_CROSSOVER_HZ,
    )

    delta_benchmark = benchmark_curve - hr_spls
    delta_power = py_pb - hr_spls
    benchmark_rms = float(np.sqrt(np.mean(delta_benchmark**2)))
    power_rms = float(np.sqrt(np.mean(delta_power**2)))

    assert benchmark_rms < 4.0
    assert float(np.max(np.abs(delta_benchmark))) < 10.0
    assert benchmark_rms < power_rms
