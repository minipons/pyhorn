#!/usr/bin/env python3
"""
Diagnose midrange gap in HiroB SPL.
Compare SPL with and without rear chamber coupling.
"""

import numpy as np
import sys, csv, copy
from pathlib import Path

REPO = Path("/Users/guillaume/P/pyhorn")
PROJECTS = REPO / "projects"
DRIVERS = REPO / "drivers"

from pyhorn_core.config.parser import parse_horn_project, parse_driver_specs
from pyhorn_core.pyhorn_physics.orchestrators import horn_response
from pyhorn_core.pyhorn_physics import rear_chamber_impedance, RHO, C

# Load HiroB
proj, horn = parse_horn_project(PROJECTS / "hirob.yaml")
driver = parse_driver_specs(DRIVERS / "FE166NV2.yaml")

# Benchmark driver (no spl_response, no lossy_le)
from dataclasses import replace

bench_driver = replace(driver, spl_response=None, lossy_le=False)

print("=" * 70)
print("DIAGNOSTIC: Rear Chamber Model — HiroB")
print("=" * 70)
print(f"\nHorn config:")
print(f"  enclosure_type={horn.enclosure_type}")
print(f"  throat={horn.throat_area*1e4:.1f} cm²  mouth={horn.mouth_area*1e4:.1f} cm²")
print(f"  path={horn.path_length:.3f} m  n_segments={horn.n_segments}")
print(f"  vrc={horn.vrc*1e6:.0f} ml  lrc={horn.lrc*1e3:.0f} mm  fr_rc={horn.fr_rc}")
rc = horn.rear_chamber
print(f"  rear_chamber.chamber_type={rc.chamber_type}")

print(f"\nDriver: fs={driver.fs} Hz  qts={driver.qts}  bl={driver.bl}  re={driver.re}")

# Frequency sweep
freqs = np.logspace(np.log10(15), np.log10(600), 500)

# ── Run FULL simulation with coupling chamber ───────────────────────────
result_coup = horn_response(freqs, bench_driver, horn, compute_distortion=False)

# ── Run with rear chamber DISABLED (vrc=0) ─────────────────────────────
horn_norc = copy.copy(horn)
horn_norc.vrc = 0.0
result_norc = horn_response(freqs, bench_driver, horn_norc, compute_distortion=False)

# ── Run with SEALE D-box model (chamber_type=sealed) ──────────────────
# We need to monkey-patch the rear_chamber.chamber_type in the horn copy
# Since horn.rear_chamber exists and is set as an attribute on the horn object
# (not a dataclass field), we can modify it directly
horn_sealed = copy.copy(horn)
# Change the chamber_type of the existing RearChamber object (shallow copy shares it)
# Make a new RearChamber with same params but sealed type
from pyhorn_core.config.models import RearChamber

rc_sealed = RearChamber(
    vrc=horn.vrc, lrc=horn.lrc, fr_rc=horn.fr_rc, chamber_type="sealed"
)
# Set the attribute directly (bypass the dataclass mechanism)
object.__setattr__(horn_sealed, "rear_chamber", rc_sealed)
result_sealed = horn_response(
    freqs, bench_driver, horn_sealed, compute_distortion=False
)

# Load Hornresp reference
BENCHMARK_ROOT = REPO / "tests/benchmarks/hornresp/hirob"
hr_csv = BENCHMARK_ROOT / "reference/hornresp_spl.csv"
hr_freqs, hr_spls = [], []
with open(hr_csv) as f:
    for row in csv.DictReader(f):
        hr_freqs.append(float(row["Freq (hertz)"]))
        hr_spls.append(float(row["SPL (dB)"]))
hr_freqs = np.array(hr_freqs)
hr_spls = np.array(hr_spls)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nSPL comparison at key frequencies:")
print(
    f"{'Freq':>7} | {'HR ref':>8} | {'coupling':>9} | {'sealed':>9} | {'no_rc':>9} | {'coup-HR':>8} | {'sealed-HR':>9} | {'norc-HR':>8}"
)
for target in [20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300, 400, 500]:
    i_py = int(np.argmin(np.abs(freqs - target)))
    i_hr = int(np.argmin(np.abs(hr_freqs - target)))
    print(
        f"{freqs[i_py]:7.1f} | {hr_spls[i_hr]:8.2f} | "
        f"{result_coup.spl[i_py]:9.2f} | {result_sealed.spl[i_py]:9.2f} | {result_norc.spl[i_py]:9.2f} | "
        f"{result_coup.spl[i_py]-hr_spls[i_hr]:+8.2f} | "
        f"{result_sealed.spl[i_py]-hr_spls[i_hr]:+9.2f} | "
        f"{result_norc.spl[i_py]-hr_spls[i_hr]:+8.2f}"
    )

print("\nFrequency bands summary (mean SPL):")
print(f"{'Band':>12} | {'HR ref':>8} | {'coupling':>9} | {'sealed':>9} | {'no_rc':>9}")
for lo, hi in [(15, 50), (50, 100), (100, 200), (200, 500)]:
    m = (freqs >= lo) & (freqs < hi)
    i_hr_lo = int(np.argmin(np.abs(hr_freqs - lo)))
    i_hr_hi = int(np.argmin(np.abs(hr_freqs - hi)))
    hr_band = hr_spls[i_hr_lo : i_hr_hi + 1]
    print(
        f"{lo:4d}-{hi:4d} Hz | {np.mean(hr_band):8.2f} | "
        f"{np.mean(result_coup.spl[m]):9.2f} | "
        f"{np.mean(result_sealed.spl[m]):9.2f} | "
        f"{np.mean(result_norc.spl[m]):9.2f}"
    )

print("\n" + "=" * 70)
print("Z_rc magnitude at key frequencies:")
print("=" * 70)
for f in [20, 40, 50, 60, 80, 100, 150, 200, 300]:
    Z_coup = rear_chamber_impedance(
        f,
        horn.vrc,
        horn.lrc,
        fr=horn.fr_rc,
        chamber_type="coupling",
        throat_area=horn.throat_area,
    )
    Z_sealed = rear_chamber_impedance(
        f,
        horn.vrc,
        horn.lrc,
        fr=horn.fr_rc,
        chamber_type="sealed",
        throat_area=horn.throat_area,
    )
    print(
        f"  f={f:3d} Hz: Z_coup=|{abs(Z_coup)/1e3:.0f}k  Z_sealed=|{abs(Z_sealed)/1e3:.0f}k"
    )

print("\n" + "=" * 70)
print("BAND-BY-BAND VERDICT")
print("=" * 70)
for band, lo, hi in [
    ("LF 15-50Hz", 15, 50),
    ("MF 50-100Hz", 50, 100),
    ("MID 100-200Hz", 100, 200),
    ("UPPER 200-500Hz", 200, 500),
]:
    m = (freqs >= lo) & (freqs < hi)
    i_hr_lo = int(np.argmin(np.abs(hr_freqs - lo)))
    i_hr_hi = int(np.argmin(np.abs(hr_freqs - hi)))
    hr_band = hr_spls[i_hr_lo : i_hr_hi + 1]
    d_coup = np.mean(result_coup.spl[m]) - np.mean(hr_band)
    d_sealed = np.mean(result_sealed.spl[m]) - np.mean(hr_band)
    d_norc = np.mean(result_norc.spl[m]) - np.mean(hr_band)
    print(f"\n  {band}:")
    print(
        f"    coupling: {np.mean(result_coup.spl[m]):.1f} dB (delta vs HR: {d_coup:+.1f} dB)"
    )
    print(
        f"    sealed:   {np.mean(result_sealed.spl[m]):.1f} dB (delta vs HR: {d_sealed:+.1f} dB)"
    )
    print(
        f"    no_rc:    {np.mean(result_norc.spl[m]):.1f} dB (delta vs HR: {d_norc:+.1f} dB)"
    )
    print(f"    HR ref:   {np.mean(hr_band):.1f} dB")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
ax = axes[0]
ax.plot(hr_freqs, hr_spls, "b-", lw=1.5, label="Hornresp reference", alpha=0.8)
ax.plot(
    freqs, result_coup.spl, "r-", lw=1.2, label="pyhorn coupling chamber", alpha=0.8
)
ax.plot(
    freqs, result_sealed.spl, "m-", lw=1.2, label="pyhorn sealed chamber", alpha=0.8
)
ax.plot(freqs, result_norc.spl, "g-", lw=1.2, label="pyhorn no rear chamber", alpha=0.8)
ax.set_xscale("log")
ax.set_xlabel("Freq (Hz)")
ax.set_ylabel("SPL (dB)")
ax.set_title("HiroB: Rear Chamber Model Comparison")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
ax.set_xlim(15, 500)

ax = axes[1]
i_hr = np.searchsorted(hr_freqs, freqs)
i_hr = np.clip(i_hr, 0, len(hr_spls) - 1)
hr_at_freqs = hr_spls[i_hr]
ax.plot(
    freqs, result_coup.spl - hr_at_freqs, "r-", lw=1.2, label="coupling - HR", alpha=0.8
)
ax.plot(
    freqs, result_sealed.spl - hr_at_freqs, "m-", lw=1.2, label="sealed - HR", alpha=0.8
)
ax.plot(
    freqs, result_norc.spl - hr_at_freqs, "g-", lw=1.2, label="no_rc - HR", alpha=0.8
)
ax.axhline(0, color="gray", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("Freq (Hz)")
ax.set_ylabel("delta dB")
ax.set_title("Delta vs Hornresp reference")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
ax.set_xlim(15, 500)

plt.tight_layout()
out_png = BENCHMARK_ROOT / "hirob_rear_chamber_diagnostic.png"
plt.savefig(out_png, dpi=130)
print(f"\nPlot saved: {out_png}")
