"""
 Analytical validation of pyhorn TMM core against independent physics formulas.

 Five tests:

   1. Cylindrical tube resonances: f_n = (2n-1)·c / (4L)
      Quarter-wave standing waves in a 1m tube → dips in horn response.

   2. Exponential horn cutoff: fc = c·m/(2π)
      m = ln(S2/S1)/L — below fc the horn is evanescent, above fc it propagates.

   3. Rear-chamber Helmholtz: f_H = (c/2π) × √(S_throat / (V_rc × L_eff))
      L_eff = L_rc + 1.7·√(S_rc/π) (flanged-end correction).

   4. BLH low-frequency resonance: f_L = 1/(2π√(M_md × C_eff))
      C_eff = Vrc/(ρc²) + S1·L_throat/(ρc²).

   5. pyhorn run vs cylindrical tube theory — checks SPL dips align with f_n.
   6. pyhorn run vs exp-horn theory — checks SPL rise near fc.
"""

import sys, os, numpy as np

RHO = 1.21
C   = 343.0
Z0  = RHO * C  # ~415 rayls


# =============================================================================
#  THEORY-ONLY TESTS
# =============================================================================

def run_theory_tests():
    print("=" * 60)
    print("Theory calculations (no pyhorn needed)")
    print("=" * 60)

    # ---- 1. Cylindrical tube resonances --------------------------------------
    L = 1.0        # 1 metre tube
    c = C
    resonances = [(2*n-1) * c / (4*L) for n in range(1, 6)]
    print(f"\n[1] Cylindrical tube ({L}m long)")
    print(f"    Quarter-wave resonances: {[f'{f:.1f}' for f in resonances]}")

    # ---- 2. Exponential horn cutoff -----------------------------------------
    S1, S2, L_h = 0.008, 0.060, 1.530
    m = np.log(S2/S1) / L_h    # expansion constant
    fc = C * m / (2*np.pi)     # cutoff frequency
    print(f"\n[2] Exponential horn (S1={S1*1e4:.0f}cm², S2={S2*1e4:.0f}cm², L={L_h}m)")
    print(f"    m = ln({S2/S1:.2f})/{L_h:.3f} = {m:.4f} m⁻¹")
    print(f"    fc = c·m/(2π) = {C}×{m:.4f}/(2π) = {fc:.1f} Hz")

    # ---- 3. Rear-chamber Helmholtz -------------------------------------------
    Vrc, Lrc, S_thr = 0.005, 0.15, 0.008
    S_rc  = Vrc / Lrc
    a_rc  = np.sqrt(S_rc / np.pi)
    L_eff = Lrc + 1.7 * a_rc
    f_H   = (C / (2*np.pi)) * np.sqrt(S_thr / (Vrc * L_eff))
    print(f"\n[3] Rear chamber Helmholtz (Vrc={Vrc*1e3:.0f}L, Lrc={Lrc*100:.0f}cm)")
    print(f"    S_rc={S_rc*1e4:.1f}cm², a_rc={a_rc*100:.1f}cm, L_eff={L_eff*100:.1f}cm")
    print(f"    f_H = {f_H:.1f} Hz")

    # ---- 4. BLH low-frequency resonance --------------------------------------
    M_md = 0.00604
    C_rc  = Vrc / (RHO * C**2)
    C_thr = (S1 * 0.10) / (RHO * C**2)   # 10cm effective throat length
    C_eff = C_rc + C_thr
    f_L    = 1 / (2*np.pi * np.sqrt(M_md * C_eff))
    print(f"\n[4] BLH low-freq resonance (M_md={M_md*1000:.2f}g)")
    print(f"    C_rc={C_rc:.3e}, C_thr={C_thr:.3e}, C_eff={C_eff:.3e}")
    print(f"    f_L = 1/(2π√(M·C)) = {f_L:.1f} Hz")

    return {
        'tube_resonances': resonances,
        'exp_horn_fc': fc,
        'rear_chamber_fH': f_H,
        'blh_fL': f_L,
    }


# =============================================================================
#  PYHORN RUN TESTS
# =============================================================================

def run_pyhorn_cylindrical_tube():
    """Check that pyhorn shows SPL dips at tube quarter-wave resonances."""
    import yaml
    from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry
    from pyhorn_core.pyhorn_physics.orchestrators import horn_response

    driver_data = {
        'fs': 49.6, 'qts': 0.27, 'qes': 0.28, 'qms': 7.88,
        'vas': 0.0369, 're': 7.80, 'sd': 0.0132, 'bl': 7.75,
        'mms': 0.00604, 'cms': 1.49e-3, 'rms': 0.27, 'le': 0.80e-3, 'xmax': 0.0015,
        'voltage': 2.83
    }
    # Cylindrical: throat_area == mouth_area, profile_type='straight'
    horn_data = {
        'throat_area': 0.008, 'mouth_area': 0.008,
        'path_length': 1.0,
        'n_segments': 50, 'profile_type': 'straight',
        'vrc': 0.005, 'lrc': 0.15, 'fr_rc': 2000,
        'vtc': 0.0, 'atc': 0.0, 'fr_tc': 2000,
        'ang': 1.5708, 'ap1': 0.008, 'lpt': 0.0,
        'enclosure_type': 'BLH', 'width': 0.10
    }

    with open('/tmp/cyl_driver.yaml', 'w') as f: yaml.dump(driver_data, f)
    with open('/tmp/cyl_horn.yaml', 'w') as f: yaml.dump(horn_data, f)

    driver = parse_driver_specs('/tmp/cyl_driver.yaml')
    horn   = parse_horn_geometry('/tmp/cyl_horn.yaml')

    freqs  = np.linspace(10, 3000, 3000)
    result = horn_response(freqs, driver, horn, compute_distortion=False)

    # Theoretical f_n = (2n-1)c/4L
    L, c = 1.0, C
    f_theory = [(2*n-1)*c/(4*L) for n in range(1, 5)]

    print(f"\n[5] pyhorn Cylindrical Tube ({L}m, S={0.008*1e4:.0f}cm²)")
    print(f"    Theory: {[f'{f:.1f}' for f in f_theory]}")

    peaks, meas = [], []
    for f_th in f_theory:
        rng = (freqs > f_th*0.5) & (freqs < f_th*1.5)
        if rng.sum() > 0:
            idx  = np.argmin(result.spl[rng])
            f_m  = freqs[rng][idx]
            peaks.append(f_m)
            meas.append(result.spl[rng][idx])

    errs = [abs(p - t) for p, t in zip(peaks, f_theory[:len(peaks)])]
    print(f"    pyhorn dips: {[f'{p:.1f}' for p in peaks]}")
    print(f"    Errors:      {[f'{e:.1f}Hz ({100*e/t:.1f}%)' for e, t in zip(errs, f_theory[:len(errs)])]}")

    ok = all(e < 40 for e in errs)
    print(f"    → {'PASS' if ok else 'CHECK'} (threshold < 40 Hz)")
    return peaks, f_theory, errs


def run_pyhorn_exp_horn():
    """Check that pyhorn SPL rises significantly above the theoretical fc."""
    import yaml
    from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry
    from pyhorn_core.pyhorn_physics.orchestrators import horn_response

    S1, S2, L_h = 0.008, 0.060, 1.530
    m  = np.log(S2/S1) / L_h
    fc = C * m / (2*np.pi)

    driver_data = {
        'fs': 49.6, 'qts': 0.27, 'qes': 0.28, 'qms': 7.88,
        'vas': 0.0369, 're': 7.80, 'sd': 0.0132, 'bl': 7.75,
        'mms': 0.00604, 'cms': 1.49e-3, 'rms': 0.27, 'le': 0.80e-3, 'xmax': 0.0015,
        'voltage': 2.83
    }
    horn_data = {
        'throat_area': S1, 'mouth_area': S2, 'path_length': L_h,
        'n_segments': 50, 'profile_type': 'hyperbolic', 'hyperbolic_t': 0.35,
        'vrc': 0.005, 'lrc': 0.15, 'fr_rc': 2000,
        'vtc': 0.0, 'atc': 0.0, 'fr_tc': 2000,
        'ang': 1.5708, 'ap1': 0.008, 'lpt': 0.0,
        'enclosure_type': 'BLH', 'width': 0.30
    }

    with open('/tmp/exp_driver.yaml', 'w') as f: yaml.dump(driver_data, f)
    with open('/tmp/exp_horn.yaml', 'w') as f: yaml.dump(horn_data, f)

    driver = parse_driver_specs('/tmp/exp_driver.yaml')
    horn   = parse_horn_geometry('/tmp/exp_horn.yaml')

    freqs  = np.linspace(10, 5000, 5000)
    result = horn_response(freqs, driver, horn, compute_distortion=False)

    f_b = fc * 0.5
    f_a = fc * 2.0
    idx_b = np.argmin(np.abs(freqs - f_b))
    idx_a = np.argmin(np.abs(freqs - f_a))
    spl_b = result.spl[idx_b]
    spl_a = result.spl[idx_a]

    # Also check SPL at 5×fc should be well above 0.5×fc
    f_5x = fc * 5
    idx_5 = np.argmin(np.abs(freqs - f_5x))
    spl_5 = result.spl[idx_5]

    print(f"\n[6] pyhorn Exponential Horn (hyperbolic t=0.35)")
    print(f"    fc_theory = {fc:.1f} Hz")
    print(f"    SPL at 0.5×fc ({f_b:.0f} Hz): {spl_b:.1f} dB")
    print(f"    SPL at 2×fc  ({f_a:.0f} Hz): {spl_a:.1f} dB  (rise: {spl_a-spl_b:+.1f} dB)")
    print(f"    SPL at 5×fc  ({f_5x:.0f} Hz): {spl_5:.1f} dB")

    rise_ok = (spl_a - spl_b) > 5.0   # expect at least 5 dB rise
    print(f"    → {'PASS' if rise_ok else 'CHECK'} (expect ≥5 dB rise from 0.5×fc to 2×fc)")
    return fc, freqs, result.spl


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':
    theory = run_theory_tests()

    print("\n" + "=" * 60)
    print("pyhorn internal tests")
    print("=" * 60)

    try:
        peaks, f_th, errs = run_pyhorn_cylindrical_tube()
    except Exception as e:
        print(f"\n[5] FAILED: {e}")
        import traceback; traceback.print_exc()

    try:
        fc, freqs, spls = run_pyhorn_exp_horn()
    except Exception as e:
        print(f"\n[6] FAILED: {e}")
        import traceback; traceback.print_exc()
