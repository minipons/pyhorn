#!/usr/bin/env python3
"""
Compare pyhorn vs Hornresp SPL for HiroB horn.
Also test Levine/Inglis vs Anechoic mouth termination.
Also test n_segments convergence.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import csv

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyhorn_core.config.parser import parse_horn_project, parse_driver_specs
from pyhorn_core.pyhorn_physics.orchestrators import horn_response
from pyhorn_core.pyhorn_physics import RHO, C
import pyhorn_core.pyhorn_physics.orchestrators as _orch
from dataclasses import replace

BENCHMARK_ROOT = REPO / "tests/benchmarks/hornresp/hirob"
HR_CSV = BENCHMARK_ROOT / "reference/hornresp_spl.csv"
PROJECT = BENCHMARK_ROOT / "fixture/horn.yaml"
DRIVER = BENCHMARK_ROOT / "fixture/driver.yaml"


def load_hornresp(path):
    freqs, spls = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            freqs.append(float(row["Freq (hertz)"]))
            spls.append(float(row["SPL (dB)"]))
    return np.array(freqs), np.array(spls)


def run_benchmark(n_segments=126, mouth_rad="anechoic", driver_overrides=None):
    driver = parse_driver_specs(DRIVER)
    proj, horn = parse_horn_project(PROJECT)

    # Override
    horn = replace(horn, n_segments=n_segments, mouth_radiation=mouth_rad)

    if driver_overrides:
        driver = replace(driver, **driver_overrides)

    freqs = np.logspace(np.log10(10), np.log10(20000), 800)
    result = horn_response(freqs, driver, horn, compute_distortion=False)
    return freqs, result


def main():
    hr_freqs, hr_spls = load_hornresp(HR_CSV)

    # Load hornresp params for reference
    with open(BENCHMARK_ROOT / "reference/hornresp_params.txt") as f:
        params = f.read()

    print("Hornresp params (key values):")
    for line in params.splitlines():
        if any(
            k in line
            for k in ["S1", "S2", "AT", "Vrc", "Lrc", "Sd", "Ang", "End Correction"]
        ):
            print(f"  {line.strip()}")

    print("\n=== Test 1: n_segments convergence ===")
    for n_seg in [50, 100, 126, 200, 400]:
        f, r = run_benchmark(
            n_segments=n_seg,
            mouth_rad="anechoic",
            driver_overrides=dict(spl_response=None, lossy_le=False),
        )
        log_f = np.log10(f)
        log_hr = np.log10(hr_freqs)
        py_spl = interp1d(
            log_f, r.spl_power_based, kind="linear", fill_value="extrapolate"
        )(log_hr)
        valid = (hr_freqs >= f.min()) & (hr_freqs <= f.max())
        d = py_spl[valid] - hr_spls[valid]
        print(
            f"  n_seg={n_seg:4d}: mean={d.mean():+6.2f}  std={d.std():5.2f}  "
            f"LF(10-100)={d[valid & (hr_freqs<100)].mean():+6.2f}  "
            f"HF(1k-10k)={d[valid & (hr_freqs>=1000) & (hr_freqs<10000)].mean():+6.2f}"
        )

    print("\n=== Test 2: Levine vs Anechoic (n_seg=126) ===")
    for mouth_rad in ["anechoic", "levine"]:
        f, r = run_benchmark(
            n_segments=126,
            mouth_rad=mouth_rad,
            driver_overrides=dict(spl_response=None, lossy_le=False),
        )
        log_f = np.log10(f)
        log_hr = np.log10(hr_freqs)
        py_spl = interp1d(
            log_f, r.spl_power_based, kind="linear", fill_value="extrapolate"
        )(log_hr)
        valid = (hr_freqs >= f.min()) & (hr_freqs <= f.max())
        d = py_spl[valid] - hr_spls[valid]
        print(f"  mouth_rad={mouth_rad:8s}: mean={d.mean():+6.2f}  std={d.std():5.2f}")
        for lo, hi in [(10, 100), (100, 1000), (1000, 5000), (5000, 10000)]:
            m = valid & (hr_freqs >= lo) & (hr_freqs < hi)
            if m.sum() > 0:
                print(
                    f"    {lo:4d}-{hi:5d} Hz: mean={d[m].mean():+6.2f}  std={d[m].std():5.2f}"
                )

    print("\n=== Test 3: Compare at key frequencies ===")
    f, r = run_benchmark(
        n_segments=126,
        mouth_rad="anechoic",
        driver_overrides=dict(spl_response=None, lossy_le=False),
    )
    log_f = np.log10(f)
    log_hr = np.log10(hr_freqs)
    py_pb = interp1d(log_f, r.spl_power_based, kind="linear", fill_value="extrapolate")(
        log_hr
    )

    f2, r2 = run_benchmark(
        n_segments=126,
        mouth_rad="levine",
        driver_overrides=dict(spl_response=None, lossy_le=False),
    )
    py_pb2 = interp1d(
        np.log10(f2), r2.spl_power_based, kind="linear", fill_value="extrapolate"
    )(log_hr)

    print(
        f"  {'Freq':>7} | {'Hornresp':>8} | {'anechoic':>8} | {'levine':>8} | {'Δ anechoic':>9} | {'Δ levine':>8}"
    )
    for target in [20, 30, 50, 80, 100, 200, 500, 1000, 2000, 3000, 5000, 8000, 10000]:
        i = np.argmin(np.abs(hr_freqs - target))
        print(
            f"  {hr_freqs[i]:7.1f} | {hr_spls[i]:8.2f} | {py_pb[i]:8.2f} | {py_pb2[i]:8.2f} | "
            f"{py_pb[i]-hr_spls[i]:+9.2f} | {py_pb2[i]-hr_spls[i]:+8.2f}"
        )

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    ax = axes[0]
    ax.plot(hr_freqs, hr_spls, "b-", lw=1.5, label="Hornresp", alpha=0.8)
    ax.plot(f, r.spl_power_based, "r-", lw=1.0, label="pyhorn anechoic", alpha=0.7)
    ax.plot(f2, r2.spl_power_based, "g--", lw=1.0, label="pyhorn levine", alpha=0.7)
    ax.set_xscale("log")
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_title("HiroB: Hornresp vs pyhorn (n_seg=126)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax = axes[1]
    valid = (hr_freqs >= f.min()) & (hr_freqs <= f.max())
    ax.plot(
        hr_freqs[valid],
        py_pb[valid] - hr_spls[valid],
        "r-",
        lw=1.0,
        label="Δ anechoic",
        alpha=0.7,
    )
    ax.plot(
        hr_freqs[valid],
        py_pb2[valid] - hr_spls[valid],
        "g-",
        lw=1.0,
        label="Δ levine",
        alpha=0.7,
    )
    ax.axhline(0, color="gray", lw=0.8)
    ax.axhline(+3, color="orange", ls="--", lw=0.5, alpha=0.5)
    ax.axhline(-3, color="orange", ls="--", lw=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Freq (Hz)")
    ax.set_ylabel("delta dB")
    ax.set_title("Delta vs Hornresp")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    out = BENCHMARK_ROOT / "hirob_mouth_rad_comparison.png"
    plt.savefig(out, dpi=130)
    print(f"\nPlot saved: {out}")


if __name__ == "__main__":
    main()
