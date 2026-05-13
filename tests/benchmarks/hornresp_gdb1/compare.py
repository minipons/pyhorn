#!/usr/bin/env python3
"""
Compare Hornresp CSV export against pyhorn simulation for GdB1 parameters.
Run from repo root: python tests/benchmarks/hornresp_gdb1/compare.py
"""
import sys, os
import csv
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Setup paths
REPO = "/Users/guillaume/P/GdB1"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/pyhorn_core")
sys.path.insert(0, f"{REPO}/pyhorn_api")
sys.path.insert(0, f"{REPO}/pyhorn_cli")

from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry
from pyhorn_core.pyhorn_physics.orchestrators import horn_response

# ── Load Hornresp CSV ─────────────────────────────────────────────────────────
hornresp_csv = f"{REPO}/tests/benchmarks/hornresp_gdb1/hornresp_spl.csv"
hr_freqs, hr_spls = [], []
with open(hornresp_csv) as f:
    reader = csv.DictReader(f)
    for row in reader:
        hr_freqs.append(float(row["Freq (hertz)"]))
        hr_spls.append(float(row["SPL (dB)"]))
hr_freqs = np.array(hr_freqs)
hr_spls = np.array(hr_spls)
print(f"Hornresp: {len(hr_freqs)} points, {hr_freqs.min():.1f}–{hr_freqs.max():.0f} Hz, SPL range {hr_spls.min():.1f}–{hr_spls.max():.1f} dB")

# ── Load and split the combined YAML ─────────────────────────────────────────
combined_yaml = f"{REPO}/tests/benchmarks/hornresp_gdb1/gdb1_hornresp.yaml"
with open(combined_yaml) as f:
    params = yaml.safe_load(f)

# DriverSpecs fields (only these go into driver YAML)
DRIVER_FIELDS = {'fs', 'qts', 'qes', 'qms', 'vas', 're', 'sd', 'bl',
                 'mms', 'cms', 'rms', 'le', 'xmax', 'voltage', 'alpha_re',
                 'le_freq_dependency', 'le_f_ref', 'lossy_le', 'le_R_e_eddy',
                 'le_f_lossy_ref', 'sensitivity_db'}
# HornGeometry fields (only these go into horn YAML)
HORN_FIELDS = {'throat_area', 'mouth_area', 'path_length', 'enclosure_type',
                'path_diff', 'ang', 'vrc', 'lrc', 'fr_rc', 'vented_box',
                'passive_radiator', 'slavbas', 'vtc', 'atc', 'fr_tc',
                'ap1', 'lpt', 'throat_adapter_type', 'profile_type',
                'hyperbolic_t', 'n_segments', 'width', 'sections',
                'conical_segments', 'rectangular_segments', 'coordinates',
                'enclosure_dims', 'driver_coord', 'discretisation', 'bend_angles',
                'rear_chamber',  # nested dict: {vrc, lrc, fr_rc, fr_tuning, chamber_type}
                'lem_step_model', 'lem_step_strength', 'lem_step_resistance',
                'segments', 'bends'}

driver_data = {k: v for k, v in params.items() if k in DRIVER_FIELDS}
horn_data = {k: v for k, v in params.items() if k in HORN_FIELDS}

driver_yaml_path = f"{REPO}/tests/benchmarks/hornresp_gdb1/gdb1_driver_only.yaml"
horn_yaml_path = f"{REPO}/tests/benchmarks/hornresp_gdb1/gdb1_horn_only.yaml"
with open(driver_yaml_path, 'w') as f:
    yaml.dump(driver_data, f)
with open(horn_yaml_path, 'w') as f:
    yaml.dump(horn_data, f)

driver = parse_driver_specs(driver_yaml_path)
horn = parse_horn_geometry(horn_yaml_path)
print(f"Driver: fs={driver.fs}, qts={driver.qts:.3f}, sd={driver.sd:.4f} m², bl={driver.bl}")
print(f"  mms={driver.mms*1000:.3f}g, cms={driver.cms*1e3:.4f} mm/N, rms={driver.rms}")
print(f"  re={driver.re}, le={driver.le*1000:.2f}mH, xmax={driver.xmax*1000:.1f}mm")
print(f"Horn: throat={horn.throat_area*1e4:.1f}cm², mouth={horn.mouth_area*1e4:.1f}cm², path={horn.path_length:.3f}m")
print(f"  profile={horn.profile_type}, hyperbolic_t={horn.hyperbolic_t}")

# ── Run pyhorn simulation ─────────────────────────────────────────────────────
py_freqs = np.linspace(10, 20000, 2000)
print(f"\nRunning pyhorn simulation ({len(py_freqs)} freq points)...")
result = horn_response(py_freqs, driver, horn, compute_distortion=False)
print(f"  SPL range: {result.spl.min():.1f}–{result.spl.max():.1f} dB")

# ── Interpolate pyhorn to Hornresp frequencies ──────────────────────────────────
log_hr_freqs = np.log10(hr_freqs)
log_py_freqs = np.log10(py_freqs)
valid = (hr_freqs >= py_freqs.min()) & (hr_freqs <= py_freqs.max())
interp_spl = interp1d(log_py_freqs, result.spl, kind='linear', fill_value='extrapolate')
interp_spl_pb = interp1d(log_py_freqs, result.spl_power_based, kind='linear', fill_value='extrapolate')
py_spl_interp = interp_spl(log_hr_freqs)
py_spl_pb_interp = interp_spl_pb(log_hr_freqs)

# ── Compute deltas ────────────────────────────────────────────────────────────
valid_mask = valid
# Compare acoustic-power-based SPL (spl_power_based) against Hornresp dB/W/m reference
# since Hornresp's SPL column is already normalized to dB/W/m (acoustic power reference)
delta_spl = py_spl_interp[valid_mask] - hr_spls[valid_mask]
delta_spl_pb = py_spl_pb_interp[valid_mask] - hr_spls[valid_mask]
print(f"\nComparison ({valid_mask.sum()} matching points):")
print(f"  SPL (pressure-based) vs Hornresp dB/W/m: mean={np.mean(delta_spl):+.2f} dB, std={np.std(delta_spl):.2f} dB")
print(f"  SPL_power_based (acoustic-power-based) vs Hornresp dB/W/m: mean={np.mean(delta_spl_pb):+.2f} dB, std={np.std(delta_spl_pb):.2f} dB")
delta = delta_spl_pb  # Use acoustic-power-based SPL for plotting

# ── Plot comparison ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

ax = axes[0]
ax.plot(hr_freqs, hr_spls, 'b-', linewidth=1.5, label='Hornresp dB/W/m', alpha=0.8)
ax.plot(py_freqs, result.spl, 'r--', linewidth=1.0, label='pyhorn (pressure-based)', alpha=0.5)
ax.plot(py_freqs, result.spl_power_based, 'g-', linewidth=1.0, label='pyhorn (acoustic-power-based)', alpha=0.8)
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('SPL (dB)')
ax.set_title('GdB1 — Hornresp vs pyhorn SPL Comparison')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

ax = axes[1]
ax.plot(hr_freqs[valid_mask], delta_spl, 'r-', linewidth=1, alpha=0.5, label='SPL (pressure-based) delta')
ax.plot(hr_freqs[valid_mask], delta_spl_pb, 'g-', linewidth=1, alpha=0.8, label='SPL_power_based delta')
ax.axhline(0, color='gray', linewidth=0.8)
ax.axhline(2, color='r', linewidth=0.5, linestyle='--', alpha=0.5)
ax.axhline(-2, color='r', linewidth=0.5, linestyle='--', alpha=0.5)
ax.set_xscale('log')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Δ SPL (pyhorn − Hornresp) dB')
ax.set_title(f' acoustic-power-based: mean={np.mean(delta_spl_pb):+.2f} dB, std={np.std(delta_spl_pb):.2f} dB')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
out_path = f"{REPO}/tests/benchmarks/hornresp_gdb1/compare_plot.png"
plt.savefig(out_path, dpi=150)
print(f"\nPlot saved: {out_path}")

# ── Per-decade breakdown ──────────────────────────────────────────────────────
print("\nPer-decade delta:")
for decade_start in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
    decade_mask = valid_mask & (hr_freqs >= decade_start) & (hr_freqs < decade_start * 10)
    if decade_mask.sum() > 0:
        d = delta[decade_mask]
        print(f"  {decade_start:6d}–{decade_start*10:6d} Hz: mean={np.mean(d):+.2f} dB, std={np.std(d):.2f} dB, n={decade_mask.sum()}")

# ── CRIT-1 diagnostic: rear chamber parameters ───────────────────────────────
print("\n── CRIT-1 diagnostic: rear chamber ──────────────────────────────────────")
print(f"  horn.vrc={horn.vrc:.5f} m³  horn.lrc={horn.lrc:.4f} m  horn.fr_rc={horn.fr_rc}")
rc = horn.rear_chamber
print(f"  horn.rear_chamber: vrc={rc.vrc:.5f} lrc={rc.lrc:.4f} "
      f"fr_rc={rc.fr_rc} chamber_type={rc.chamber_type} fr_tuning={getattr(rc, 'fr_tuning', 'N/A')}")
if horn.vrc <= 0:
    print("  ⚠ REAR CHAMBER INACTIVE: horn.vrc=0 — rear_chamber_impedance receives volume=0 → Z_ab=0")
    print("    Root cause: parse_horn_geometry() does NOT copy rear_chamber params to horn.{vrc,lrc,fr_rc}")
    print("    The orchestrator reads horn.vrc=0 (top-level) instead of horn.rear_chamber.vrc=0.005")
else:
    print("  ✓ Rear chamber active")
    # Compute port area and effective tuning frequency for diagnostics
    from pyhorn_core.pyhorn_physics import C, RHO
    vrc_m3, lrc_m = horn.vrc, horn.lrc
    fb_diag = getattr(rc, 'fr_tuning', 49.6) or 49.6
    A_port_d = (2.0 * np.pi * fb_diag / C) ** 2 * vrc_m3 * 1.5 * lrc_m
    A_port_d = max(A_port_d, 1e-6)
    for _ in range(5):
        a_pipe_d = np.sqrt(A_port_d / np.pi)
        L_eff_d = lrc_m + 0.6 * a_pipe_d
        A_port_d = (2.0 * np.pi * fb_diag / C) ** 2 * vrc_m3 * L_eff_d
        A_port_d = max(A_port_d, 1e-6)
    dia_cm = 2 * np.sqrt(A_port_d / np.pi) * 100
    print(f"    Port: Ø{dia_cm:.1f} cm  L_eff={L_eff_d*100:.1f} cm  fb≈{fb_diag:.1f} Hz")

# Show raw acoustic power at key frequencies (no sensitivity_db correction)
print("\n── Raw acoustic power (no sensitivity_db) at key frequencies ─────────────")
for f_target in [20, 30, 40, 50, 60, 80, 100, 200]:
    i_hr = np.argmin(np.abs(hr_freqs - f_target))
    j_py = np.argmin(np.abs(py_freqs - f_target))
    raw_spl = 10.0 * np.log10(max(result.acoustic_power[j_py], 1e-12) / 1e-12)
    i_delta = np.argmin(np.abs(hr_freqs[valid_mask] - f_target))
    delta_with_cal = delta_spl_pb[valid_mask][i_delta]
    print(f"  {f_target:4d} Hz: Hornresp={hr_spls[i_hr]:.1f}  "
          f"pyhorn_raw={raw_spl:.1f}  Δraw={raw_spl-hr_spls[i_hr]:+.1f}  Δw/Cal={delta_with_cal:+.1f}")
