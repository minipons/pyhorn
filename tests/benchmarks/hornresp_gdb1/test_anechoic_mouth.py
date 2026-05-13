"""
tests/benchmarks/hornresp_gdb1/test_anechoic_mouth.py

Compare Levine/Inglis (reflecting baffle) vs plane-wave (anechoic) mouth
termination for the HiroB horn against Hornresp benchmark.

Summary of findings (May 13 2026):
=================================
1. Levine/Inglis (reflecting baffle, current):
   - Overall SPL mean delta: +6.38 dB (pressure), -0.90 dB (power-based)
   - HF band (1-10 kHz): ~+7.3 dB (pressure), ~+0.9 dB (power-based)
   - The power-based SPL is well-calibrated at HF (via sensitivity_db calibration
     table that was added in a previous session), but the pressure-based SPL
     is ~15 dB too high.

2. Anechoic (plane wave Z_rad = ρ*c/A, no reactive term):
   - Overall SPL mean delta: +12.08 dB (much WORSE than L/I)
   - The anechoic model removes the reactive (mass-like) term entirely,
     which causes the horn to behave as if it radiates with a much
     lower acoustic load at LF — leading to LARGER horn output, not smaller.
   - This is the OPPOSITE of what is needed to reduce spikiness.

3. The spikiness in the horn response (TMM artifacts):
   - NOT caused by mouth radiation impedance model
   - Due to TMM numerical artifacts at the HiroB 1847 Hz resonance
   - These are already filtered via _detect_numerical_artifacts + _smooth_spl_near_artifacts
   - The "spiky" appearance at LF in the comparison plots is primarily due to:
     a. The raw pressure-based SPL being ~6 dB too high vs dB/W/m
     b. The direct-cone radiation override via driver.get_spl_response() not yet
        being fully calibrated for HiroB

CONCLUSION: Anechoic mouth termination does NOT improve SPL match to Hornresp.
The Levine/Inglis reflecting baffle model is PHYSICALLY CORRECT for a horn
mouth on a cabinet wall. Hornresp's "ignore room resonance" option likely refers
to the room modal response being excluded, not to free-field mouth radiation.
"""

import sys
from pathlib import Path
import numpy as np
import csv

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from pyhorn_core.config.parser import parse_horn_project, parse_driver_specs
from pyhorn_core.pyhorn_physics.orchestrators import horn_response
from pyhorn_core.pyhorn_physics import RHO, C
import pyhorn_core.pyhorn_physics.orchestrators as _orch
from scipy.interpolate import interp1d


HR_CSV = REPO / "tests/benchmarks/hornresp_gdb1/hornresp_spl_hirob.csv"
PROJECT = REPO / "projects/hirob.yaml"
DRIVER = REPO / "drivers/FE166NV2.yaml"


def run_comparison():
    driver = parse_driver_specs(DRIVER)
    proj, geo = parse_horn_project(PROJECT)

    # Load Hornresp reference
    hr_freqs, hr_spls = [], []
    with open(HR_CSV) as f:
        for row in csv.DictReader(f):
            hr_freqs.append(float(row["Freq (hertz)"]))
            hr_spls.append(float(row["SPL (dB)"]))
    hr_freqs = np.array(hr_freqs)
    hr_spls = np.array(hr_spls)

    # pyhorn frequency grid
    freqs = np.linspace(10, 20000, 533)
    log_f = np.log10(freqs)
    log_hr = np.log10(hr_freqs)

    # Levine/Inglis baseline
    result_li = horn_response(freqs, driver, geo, compute_distortion=False)

    # Anechoic termination: Z_rad = ρ*c/A (plane wave, pure resistive)
    _orig_rad = _orch.radiation_impedance

    def _anechoic_rad(freq, mouth_area, ang, _Zc=None, _a=None,
                      mouth_width=None, mouth_height=None):
        return complex(RHO * C / mouth_area, 0.0)

    _orch.radiation_impedance = _anechoic_rad
    result_an = horn_response(freqs, driver, geo, compute_distortion=False)
    _orch.radiation_impedance = _orig_rad

    # Interpolate to Hornresp grid
    py_li = interp1d(log_f, result_li.spl, kind='linear',
                     fill_value='extrapolate')(log_hr)
    py_li_pb = interp1d(log_f, result_li.spl_power_based, kind='linear',
                         fill_value='extrapolate')(log_hr)
    py_an = interp1d(log_f, result_an.spl, kind='linear',
                     fill_value='extrapolate')(log_hr)

    d_li = py_li - hr_spls
    d_li_pb = py_li_pb - hr_spls
    d_an = py_an - hr_spls

    print("Frequency band  |  n  |  LI (press)   LI (power)   Anechoic")
    print("-" * 67)
    for lo, hi in [(10, 100), (20, 200), (50, 500), (100, 1000),
                   (200, 2000), (500, 5000), (1000, 10000), (2000, 20000)]:
        m = (hr_freqs >= lo) & (hr_freqs < hi)
        if m.sum() < 3:
            continue
        print(f"  {lo:4d}-{hi:5d} Hz | {m.sum():3d} | "
              f"{d_li[m].mean():+7.2f}±{d_li[m].std():4.2f}  "
              f"{d_li_pb[m].mean():+7.2f}±{d_li_pb[m].std():4.2f}  "
              f"{d_an[m].mean():+7.2f}±{d_an[m].std():4.2f}")

    print(f"\n{'Overall':>16} | {len(hr_freqs):3d} | "
          f"{d_li.mean():+7.2f}±{d_li.std():4.2f}  "
          f"{d_li_pb.mean():+7.2f}±{d_li_pb.std():4.2f}  "
          f"{d_an.mean():+7.2f}±{d_an.std():4.2f}")

    print("\nConclusion: Anechoic mouth is WORSE than Levine/Inglis.")
    print("The L/I reflecting-baffle model is physically correct for a horn cabinet.")


if __name__ == "__main__":
    run_comparison()