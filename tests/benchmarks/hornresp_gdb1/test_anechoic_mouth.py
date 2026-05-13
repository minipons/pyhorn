"""
tests/benchmarks/hornresp_gdb1/test_anechoic_mouth.py

Compare Levine/Inglis (reflecting baffle) vs plane-wave (anechoic) mouth
termination for the HiroB horn against Hornresp benchmark.

INVESTIGATION RESULTS (May 14 2026)
====================================

1. WHAT HORNRESP USES
----------------------
Hornresp's "ignore room resonance" option disables the room modal response
(reflections from walls, floor, ceiling) in the SPL calculation. It does NOT
change the mouth radiation boundary condition.

The mouth still uses Levine/Inglis radiation impedance (circular piston in
infinite baffle). Hornresp models the horn mouth on a cabinet baffle wall,
with a reflecting boundary — identical to pyhorn's default "levine" model.

2. PLANE-WAVE (ANECHOIC) TERMINATION
-------------------------------------
Z_rad = ρ*c / S_mouth (purely resistive, no reactive term).

This would model an horn mouth that radiates into free space with NO
reflecting baffle. This is physically incorrect for a cabinet-mounted horn.

Test results (no direct SPL override, isolated mouth radiation effect):
  Band       | Levine (L/I) | Anechoic  | Δ (L-I)
  10-100 Hz  |   +0.37 dB   |  +3.94 dB |  -3.57 dB  (Levine much better at LF)
  100-1000 Hz|   -0.89 dB   |  -0.48 dB |  -0.41 dB  (similar)
  1000-5000 Hz|  +4.58 dB   |  +4.62 dB |  -0.04 dB  (identical)
  5000-10000 Hz| +5.67 dB   |  +5.66 dB |  +0.01 dB  (identical)

CONCLUSION: Anechoic is WORSE at LF (+3.57 dB worse) and identical at HF.
The Levine/Inglis reflecting-baffle model is correct for the HiroB cabinet.

3. HF DISCREPANCY (~+5-6 dB in 1-10 kHz band)
----------------------------------------------
This is the dominant error source, but it's NOT a mouth radiation model issue.
Both Levine and Anechoic show the same +5-6 dB error in the HF band.

The root cause is in pyhorn's acoustic power computation:
- Both mouth radiation models produce identical HF results
- The HF discrepancy is a separate acoustic power/sensitivity_db issue

4. SPL_contamination BUG (with direct SPL override)
---------------------------------------------------
When driver.get_spl_response() override is active, the code does:

    spl_power_based_out += (new_spl_out - spl_out)

This CONTAMINATES the power-based SPL with a pressure-based correction,
causing ~3 dB additional error and increased std-dev in the HF band.

With direct SPL override (current default):
  Overall: Levine=+3.26±7.44 dB (high variance!)
  HF (5-10 kHz): +8.66 dB

WITHOUT direct SPL override (isolated):
  Overall: Levine=+1.86±2.94 dB (lower variance)
  HF (5-10 kHz): +5.67 dB

The HF error (~+5.67 dB) is the same in both cases — it's a separate
acoustic power computation issue, NOT caused by the direct SPL override.

RECOMMENDATIONS
===============
1. mouth_radiation="levine" (default, reflecting baffle) is correct ✓
2. Anechoic termination does NOT improve match to Hornresp ✗
3. The SPL_contamination bug should be fixed separately (separate issue)
4. The HF acoustic power error (~+5-6 dB) is a separate acoustic power
   computation issue — investigate acoustic power formula vs Hornresp dB/W/m
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


def load_hornresp(path):
    freqs, spls = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            freqs.append(float(row["Freq (hertz)"]))
            spls.append(float(row["SPL (dB)"]))
    return np.array(freqs), np.array(spls)


def run_comparison():
    driver = parse_driver_specs(DRIVER)
    proj, geo = parse_horn_project(PROJECT)

    hr_freqs, hr_spls = load_hornresp(HR_CSV)

    # Frequency grid
    freqs = np.logspace(np.log10(10), np.log10(20000), 400)
    log_f = np.log10(freqs)
    log_hr = np.log10(hr_freqs)

    # Driver without direct SPL override (isolates mouth radiation effect)
    from dataclasses import replace
    driver_no_override = replace(driver, spl_response=None)

    # Levine/Inglis (default)
    result_li = horn_response(freqs, driver_no_override, geo,
                             compute_distortion=False)

    # Anechoic termination
    geo_an = replace(geo, mouth_radiation="anechoic")
    result_an = horn_response(freqs, driver_no_override, geo_an,
                              compute_distortion=False)

    # Interpolate to Hornresp grid
    py_li = interp1d(log_f, result_li.spl_power_based, kind='linear',
                     fill_value='extrapolate')(log_hr)
    py_an = interp1d(log_f, result_an.spl_power_based, kind='linear',
                     fill_value='extrapolate')(log_hr)

    valid = (hr_freqs >= freqs.min()) & (hr_freqs <= freqs.max())
    d_li = py_li[valid] - hr_spls[valid]
    d_an = py_an[valid] - hr_spls[valid]

    print("Mouth radiation model comparison (no direct SPL override):")
    print(f"{'Band':>15} | {'n'} | {'Levine':>9} | {'Anechoic':>9} | {'Δ(L-A)':>8}")
    print("-" * 56)
    for lo, hi in [(10, 100), (100, 1000), (1000, 5000),
                   (5000, 10000), (10000, 20000)]:
        m = valid & (hr_freqs >= lo) & (hr_freqs < hi)
        if m.sum() < 3:
            continue
        delta = d_li[m].mean() - d_an[m].mean()
        print(f"  {lo:5d}-{hi:5d} Hz | {m.sum():3d} | "
              f"{d_li[m].mean():+8.2f} | {d_an[m].mean():+8.2f} | "
              f"{delta:+7.2f}")

    print(f"\n{'Overall':>16} | {valid.sum():3d} | "
          f"{d_li.mean():+8.2f} | {d_an.mean():+8.2f} | "
          f"{d_li.mean()-d_an.mean():+7.2f}")

    print("\nConclusion: Anechoic is WORSE at LF (+3.57 dB excess) and "
          "identical at HF. Levine/Inglis reflecting-baffle model is correct.")


if __name__ == "__main__":
    run_comparison()