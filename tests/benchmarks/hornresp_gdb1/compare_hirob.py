#!/usr/bin/env python3
"""
Compare Hornresp CSV (hornresp_spl_hirob.csv) against pyhorn for the HiroB project.
Run from repo root: python tests/benchmarks/hornresp_gdb1/compare_hirob.py

This comparison intentionally disables pyhorn's productized driver layers
(`spl_response`, `lossy_le`) and compares Hornresp against pyhorn's
power-based system reference (`result.spl_power_based`).

The pressure-based total (`result.spl`) is still printed as a diagnostic for
BLH interference notches, but it is not the Hornresp benchmark target.
"""

from dataclasses import replace
import sys
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyhorn_core.config.parser import parse_horn_project, parse_driver_specs
from pyhorn_core.pyhorn_physics.orchestrators import horn_response

BENCHMARK_ROOT = REPO / "tests/benchmarks/hornresp/hirob"
HR_CSV = BENCHMARK_ROOT / "reference/hornresp_spl.csv"
PROJECT = BENCHMARK_ROOT / "fixture/horn.yaml"
DRIVER = BENCHMARK_ROOT / "fixture/driver.yaml"
OUT_PNG = BENCHMARK_ROOT / "compare_hirob_plot.png"


def _load_hornresp_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    freqs, spls = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            freqs.append(float(row["Freq (hertz)"]))
            spls.append(float(row["SPL (dB)"]))
    return np.array(freqs), np.array(spls)


def build_hirob_benchmark_driver(driver):
    return replace(driver, spl_response=None, lossy_le=False)


def _interp_log_response(
    target_freqs: np.ndarray, source_freqs: np.ndarray, response: np.ndarray
) -> np.ndarray:
    return interp1d(
        np.log10(source_freqs),
        response,
        kind="linear",
        fill_value="extrapolate",
    )(np.log10(target_freqs))


def build_hirob_reference_curves(
    hr_freqs: np.ndarray,
    py_freqs: np.ndarray,
    result,
) -> tuple[np.ndarray, np.ndarray]:
    py_spl = _interp_log_response(hr_freqs, py_freqs, result.spl)
    py_pb = _interp_log_response(hr_freqs, py_freqs, result.spl_power_based)
    return py_spl, py_pb


def main() -> None:
    hr_freqs, hr_spls = _load_hornresp_csv(HR_CSV)
    print(
        f"Hornresp: {len(hr_freqs)} pts, {hr_freqs.min():.1f}-{hr_freqs.max():.0f} Hz, "
        f"SPL {hr_spls.min():.1f}-{hr_spls.max():.1f} dB"
    )

    proj, horn = parse_horn_project(PROJECT)
    driver = parse_driver_specs(DRIVER)
    benchmark_driver = build_hirob_benchmark_driver(driver)
    print(f"Project: {proj.name}  enc={horn.enclosure_type}  ang={horn.ang}")
    print(
        f"Driver fs={benchmark_driver.fs} qts={benchmark_driver.qts} "
        f"sd={benchmark_driver.sd:.4f}m^2 bl={benchmark_driver.bl} re={benchmark_driver.re}"
    )
    print(
        f"Horn throat={horn.throat_area*1e4:.1f}cm^2 mouth={horn.mouth_area*1e4:.1f}cm^2 "
        f"L={horn.path_length:.3f}m profile={horn.profile_type} T={horn.hyperbolic_t}"
    )
    print(
        f"Benchmark overrides: spl_response={'off' if benchmark_driver.spl_response is None else 'on'}  "
        f"lossy_le={'on' if benchmark_driver.lossy_le else 'off'}"
    )

    py_freqs = np.logspace(
        np.log10(max(hr_freqs.min(), 10)),
        np.log10(min(hr_freqs.max(), 20000)),
        1500,
    )
    result = horn_response(py_freqs, benchmark_driver, horn, compute_distortion=False)
    print(
        f"pyhorn: spl {result.spl.min():.1f}-{result.spl.max():.1f}  "
        f"spl_power_based {result.spl_power_based.min():.1f}-{result.spl_power_based.max():.1f}"
    )

    valid = (hr_freqs >= py_freqs.min()) & (hr_freqs <= py_freqs.max())
    py_spl, py_pb = build_hirob_reference_curves(hr_freqs, py_freqs, result)

    d_pressure = py_spl[valid] - hr_spls[valid]
    d_power = py_pb[valid] - hr_spls[valid]
    print("\nOverall delta vs Hornresp:")
    print(
        f"  spl_power_based:     mean={d_power.mean():+6.2f} dB  std={d_power.std():5.2f} dB"
    )
    print(
        f"  spl (pressure):      mean={d_pressure.mean():+6.2f} dB  std={d_pressure.std():5.2f} dB"
    )

    print("\nPer-decade delta (spl_power_based vs Hornresp):")
    print(f"  {'band':>14}  {'n':>4}  {'mean':>7}  {'std':>5}  {'max|d|':>7}")
    for lo in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
        m = valid & (hr_freqs >= lo) & (hr_freqs < lo * 10)
        if m.sum() < 3:
            continue
        d = py_pb[m] - hr_spls[m]
        print(
            f"  {lo:5}-{lo*10:5} Hz  {m.sum():4d}  {d.mean():+7.2f}  {d.std():5.2f}  {np.max(np.abs(d)):7.2f}"
        )

    notch_target = 196.0
    notch_idx = int(np.argmin(np.abs(hr_freqs - notch_target)))
    print("\nBLH notch diagnostic:")
    print(
        f"  {hr_freqs[notch_idx]:7.1f} Hz | HR {hr_spls[notch_idx]:7.2f} | "
        f"pyhorn-spl {py_spl[notch_idx]:7.2f} | pyhorn-pb {py_pb[notch_idx]:7.2f}"
    )
    print(
        f"            direct {result.direct_spl[np.argmin(np.abs(py_freqs - hr_freqs[notch_idx]))]:7.2f} | "
        f"horn {result.horn_spl[np.argmin(np.abs(py_freqs - hr_freqs[notch_idx]))]:7.2f}"
    )

    print("\nPoint comparison at key frequencies:")
    print(
        f"  {'freq':>7} | {'HR':>7} | {'pyhorn-pb':>10} | {'pyhorn-spl':>10} | {'d_power':>8}"
    )
    for target in [20, 30, 50, 80, 100, 196, 200, 500, 1000, 2000, 5000]:
        i = int(np.argmin(np.abs(hr_freqs - target)))
        print(
            f"  {hr_freqs[i]:7.1f} | {hr_spls[i]:7.2f} | {py_pb[i]:10.2f} | {py_spl[i]:10.2f} | {py_pb[i] - hr_spls[i]:+8.2f}"
        )

    print("\nCRIT-1 diagnostic (rear chamber):")
    print(
        f"  horn.vrc={horn.vrc:.5f} m^3  horn.lrc={horn.lrc:.4f} m  horn.fr_rc={horn.fr_rc}"
    )
    rc = horn.rear_chamber
    if rc is not None:
        print(
            f"  rear_chamber: vrc={rc.vrc:.5f}  lrc={rc.lrc:.4f}  fr_rc={rc.fr_rc}  "
            f"chamber_type={rc.chamber_type}"
        )
        if rc.chamber_type == "sealed" and rc.lrc > 0:
            f_qw = 343.0 / (4.0 * rc.lrc)
            print(
                f"  WARNING: chamber_type=sealed creates spurious quarter-wave resonance at "
                f"c/(4*Lrc)={f_qw:.0f} Hz"
            )
            print(
                f"           For BLH the documented-correct model is chamber_type=coupling "
                f"(see pyhorn_core/CRIT1_calibration_analysis.md)"
            )

    print("\nDriver calibration:")
    sd = benchmark_driver.sensitivity_db
    if isinstance(sd, np.ndarray) and sd.size > 0:
        print(f"  sensitivity_db: {sd.tolist()}")
    else:
        print(
            f"  sensitivity_db: {sd}  (no calibration table — spl_power_based is raw)"
        )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    ax = axes[0]
    ax.plot(hr_freqs, hr_spls, "b-", lw=1.5, label="Hornresp", alpha=0.8)
    ax.plot(
        hr_freqs,
        py_pb,
        color="#f59e0b",
        lw=1.2,
        label="pyhorn spl_power_based (benchmark)",
    )
    ax.plot(
        py_freqs, result.spl, "r--", lw=0.8, label="pyhorn spl (pressure)", alpha=0.45
    )
    ax.plot(
        py_freqs,
        result.spl_power_based,
        "g-",
        lw=0.9,
        label="pyhorn spl_power_based",
        alpha=0.65,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_title(f"HiroB — Hornresp vs pyhorn spl_power_based  ({proj.name})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    ax.plot(
        hr_freqs[valid],
        d_power,
        color="#f59e0b",
        lw=1.0,
        label="pyhorn-pb - HR",
    )
    ax.plot(
        hr_freqs[valid], d_pressure, "r-", lw=0.8, alpha=0.4, label="pyhorn-spl - HR"
    )
    ax.plot(hr_freqs[valid], d_power, "g-", lw=0.8, alpha=0.55, label="pyhorn-pb - HR")
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(+2, color="#f59e0b", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-2, color="#f59e0b", ls="--", lw=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("delta dB")
    ax.set_title(
        f"spl_power_based vs HR: mean={d_power.mean():+.2f} std={d_power.std():.2f}"
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\nplot saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
