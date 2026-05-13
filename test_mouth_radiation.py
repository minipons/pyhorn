"""
Debug: verify that the mouth radiation model patch is actually working.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from pyhorn_core.config.parser import parse_horn_project, parse_driver_specs
from pyhorn_core.pyhorn_physics import RHO, C, radiation_impedance

driver = parse_driver_specs('drivers/FE166NV2.yaml')
proj, geo = parse_horn_project('projects/hirob.yaml')

freqs = np.linspace(20, 2000, 500)
mouth_area = geo.mouth_area  # 0.08 m²
mouth_radius = np.sqrt(mouth_area / np.pi)
ang = geo.ang  # half-space

Zc_val = RHO * C

# ── Verify patch works ──────────────────────────────────────────────────────
print("BEFORE patch:")
for f in [100, 500, 1000]:
    z = radiation_impedance(f, mouth_area, ang)
    print(f"  f={f}: Zrad = {z}")

# Monkey-patch
import pyhorn_core.pyhorn_physics as _phys

_orig_rad = _phys.radiation_impedance

def _plane_wave_rad(f, mouth_area, ang, _Zc=None, _a=None, mouth_width=None, mouth_height=None):
    Zc_val = RHO * C
    return complex(Zc_val / mouth_area, 0.0)  # pure resistive

_phys.radiation_impedance = _plane_wave_rad

print("\nAFTER patch:")
for f in [100, 500, 1000]:
    z = _phys.radiation_impedance(f, mouth_area, ang)
    print(f"  f={f}: Zrad = {z}")

# ── Check horn_response internals ──────────────────────────────────────────
# Import the internal function and call with verbose debug
from pyhorn_core.pyhorn_physics import radiation_impedance as _rad_check

print("\nDirect import check after patch:")
for f in [100, 500, 1000]:
    z = _rad_check(f, mouth_area, ang)
    print(f"  f={f}: Zrad = {z}")

# ── Full horn_response test ─────────────────────────────────────────────────
from pyhorn_core.solver.models import horn_response

print("\nRunning horn_response with patched radiation_impedance...")
result = horn_response(freqs, driver, geo)

print("SPL at select frequencies:")
for f in [20, 50, 100, 200, 500, 1000, 2000]:
    idx = np.argmin(np.abs(freqs - f))
    print(f"  {f}Hz: SPL={result.spl[idx]:.1f}")

# Restore
_phys.radiation_impedance = _orig_rad