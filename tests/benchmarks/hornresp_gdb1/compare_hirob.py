#!/usr/bin/env python3
"""
Compare Hornresp CSV (hornresp_spl_hirob.csv) against pyhorn for the HiroB project.
Run from repo root: python tests/benchmarks/hornresp_gdb1/compare_hirob.py
"""
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

HR_CSV = REPO / "tests/benchmarks/hornresp_gdb1/hornresp_spl_hirob.csv"
PROJECT = REPO / "projects/hirob.yaml"
DRIVER = REPO / "drivers/FE166NV2.yaml"
OUT_PNG = REPO / "tests/benchmarks/hornresp_gdb1/compare_hirob_plot.png"

hr_freqs, hr_spls = [], []
with open(HR_CSV) as f:
    for row in csv.DictReader(f):
        hr_freqs.append(float(row["Freq (hertz)"]))
        hr_spls.append(float(row["SPL (dB)"]))
hr_freqs = np.array(hr_freqs)
hr_spls = np.array(hr_spls)
print(f"Hornresp: {len(hr_freqs)} pts, {hr_freqs.min():.1f}-{hr_freqs.max():.0f} Hz, "
      f"SPL {hr_spls.min():.1f}-{hr_spls.max():.1f} dB")

proj, horn = parse_horn_project(PROJECT)
driver = parse_driver_specs(DRIVER)
print(f"Project: {proj.name}  enc={horn.enclosure_type}  ang={horn.ang}")
print(f"Driver fs={driver.fs} qts={driver.qts} sd={driver.sd:.4f}m^2 bl={driver.bl} re={driver.re}")
print(f"Horn throat={horn.throat_area*1e4:.1f}cm^2 mouth={horn.mouth_area*1e4:.1f}cm^2 "
      f"L={horn.path_length:.3f}m profile={horn.profile_type} T={horn.hyperbolic_t}")

py_freqs = np.logspace(np.log10(max(hr_freqs.min(), 10)), np.log10(min(hr_freqs.max(), 20000)), 1500)
result = horn_response(py_freqs, driver, horn, compute_distortion=False)
print(f"pyhorn: spl {result.spl.min():.1f}-{result.spl.max():.1f}  "
      f"spl_power_based {result.spl_power_based.min():.1f}-{result.spl_power_based.max():.1f}")

log_hr, log_py = np.log10(hr_freqs), np.log10(py_freqs)
valid = (hr_freqs >= py_freqs.min()) & (hr_freqs <= py_freqs.max())
py_spl = interp1d(log_py, result.spl, kind="linear", fill_value="extrapolate")(log_hr)
py_pb = interp1d(log_py, result.spl_power_based, kind="linear", fill_value="extrapolate")(log_hr)

d_pressure = py_spl[valid] - hr_spls[valid]
d_power = py_pb[valid] - hr_spls[valid]
print(f"\nOverall delta vs Hornresp dB/W/m (correct ref: spl_power_based):")
print(f"  spl (pressure):     mean={d_pressure.mean():+6.2f} dB  std={d_pressure.std():5.2f} dB")
print(f"  spl_power_based:    mean={d_power.mean():+6.2f} dB  std={d_power.std():5.2f} dB")

print("\nPer-decade delta (spl_power_based vs Hornresp):")
print(f"  {'band':>14}  {'n':>4}  {'mean':>7}  {'std':>5}  {'max|d|':>7}")
for lo in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    m = valid & (hr_freqs >= lo) & (hr_freqs < lo * 10)
    if m.sum() < 3:
        continue
    d = py_pb[m] - hr_spls[m]
    print(f"  {lo:5}-{lo*10:5} Hz  {m.sum():4d}  {d.mean():+7.2f}  {d.std():5.2f}  {np.max(np.abs(d)):7.2f}")

print("\nPoint comparison at key frequencies:")
print(f"  {'freq':>7} | {'HR':>7} | {'pyhorn-spl':>10} | {'pyhorn-pb':>10} | {'d_press':>7} | {'d_power':>7}")
for target in [20, 30, 50, 80, 100, 200, 500, 1000, 2000, 5000]:
    i = int(np.argmin(np.abs(hr_freqs - target)))
    print(f"  {hr_freqs[i]:7.1f} | {hr_spls[i]:7.2f} | {py_spl[i]:10.2f} | {py_pb[i]:10.2f} | "
          f"{py_spl[i]-hr_spls[i]:+7.2f} | {py_pb[i]-hr_spls[i]:+7.2f}")

print("\nCRIT-1 diagnostic (rear chamber):")
print(f"  horn.vrc={horn.vrc:.5f} m^3  horn.lrc={horn.lrc:.4f} m  horn.fr_rc={horn.fr_rc}")
rc = horn.rear_chamber
if rc is not None:
    print(f"  rear_chamber: vrc={rc.vrc:.5f}  lrc={rc.lrc:.4f}  fr_rc={rc.fr_rc}  "
          f"chamber_type={rc.chamber_type}")
    if rc.chamber_type == "sealed" and rc.lrc > 0:
        f_qw = 343.0 / (4.0 * rc.lrc)
        print(f"  WARNING: chamber_type=sealed creates spurious quarter-wave resonance at "
              f"c/(4*Lrc)={f_qw:.0f} Hz")
        print(f"           For BLH the documented-correct model is chamber_type=coupling "
              f"(see pyhorn_core/CRIT1_calibration_analysis.md)")

print("\nDriver calibration:")
sd = driver.sensitivity_db
if isinstance(sd, np.ndarray) and sd.size > 0:
    print(f"  sensitivity_db: {sd.tolist()}")
else:
    print(f"  sensitivity_db: {sd}  (no calibration table — spl_power_based is raw)")

fig, axes = plt.subplots(2, 1, figsize=(14, 9))
ax = axes[0]
ax.plot(hr_freqs, hr_spls, "b-", lw=1.5, label="Hornresp dB/W/m", alpha=0.8)
ax.plot(py_freqs, result.spl, "r--", lw=0.8, label="pyhorn spl (pressure)", alpha=0.5)
ax.plot(py_freqs, result.spl_power_based, "g-", lw=1.0, label="pyhorn spl_power_based", alpha=0.85)
ax.set_xscale("log"); ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("SPL (dB)")
ax.set_title(f"HiroB — Hornresp vs pyhorn  ({proj.name})")
ax.grid(True, which="both", alpha=0.3); ax.legend()

ax = axes[1]
ax.plot(hr_freqs[valid], d_pressure, "r-", lw=0.8, alpha=0.5, label="pyhorn-spl - HR")
ax.plot(hr_freqs[valid], d_power, "g-", lw=1.0, alpha=0.85, label="pyhorn-pb - HR")
ax.axhline(0, color="gray", lw=0.8)
ax.axhline(+2, color="r", ls="--", lw=0.5, alpha=0.5)
ax.axhline(-2, color="r", ls="--", lw=0.5, alpha=0.5)
ax.set_xscale("log"); ax.set_xlabel("Freq (Hz)"); ax.set_ylabel("delta dB")
ax.set_title(f"power-based vs HR: mean={d_power.mean():+.2f} std={d_power.std():.2f}")
ax.grid(True, which="both", alpha=0.3); ax.legend()

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=130)
print(f"\nplot saved: {OUT_PNG}")
