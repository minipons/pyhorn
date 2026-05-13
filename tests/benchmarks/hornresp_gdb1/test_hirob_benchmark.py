import csv
from pathlib import Path

import numpy as np

from pyhorn_core.config.parser import parse_horn_project, parse_driver_specs
from pyhorn_core.pyhorn_physics.orchestrators import horn_response
from tests.benchmarks.hornresp_gdb1.compare_hirob import (
    build_hirob_benchmark_driver,
    build_hirob_reference_curves,
)

REPO = Path(__file__).resolve().parents[3]
BENCHMARK_ROOT = REPO / "tests/benchmarks/hornresp/hirob"
HR_CSV = BENCHMARK_ROOT / "reference/hornresp_spl.csv"
PROJECT = BENCHMARK_ROOT / "fixture/horn.yaml"
DRIVER = BENCHMARK_ROOT / "fixture/driver.yaml"


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


def test_hirob_power_based_tracks_reference_within_bounds():
    hr_freqs, hr_spls = _load_hornresp_csv()
    _, horn = parse_horn_project(PROJECT)
    driver = build_hirob_benchmark_driver(parse_driver_specs(DRIVER))

    py_freqs = np.logspace(
        np.log10(max(hr_freqs.min(), 10.0)),
        np.log10(min(hr_freqs.max(), 20000.0)),
        1500,
    )
    result = horn_response(py_freqs, driver, horn, compute_distortion=False)
    py_spl, py_pb = build_hirob_reference_curves(hr_freqs, py_freqs, result)

    delta_pressure = py_spl - hr_spls
    delta_power = py_pb - hr_spls
    pressure_rms = float(np.sqrt(np.mean(delta_pressure**2)))
    power_rms = float(np.sqrt(np.mean(delta_power**2)))
    notch_idx = int(np.argmin(np.abs(hr_freqs - 196.0)))

    assert power_rms < 4.0
    assert float(np.max(np.abs(delta_power))) < 6.5
    assert power_rms < pressure_rms
    assert abs(delta_power[notch_idx]) < abs(delta_pressure[notch_idx])
