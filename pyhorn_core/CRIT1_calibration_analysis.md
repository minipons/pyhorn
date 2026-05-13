# CRIT-1 Calibration Analysis: Rear Chamber V_rc / l_rc

**Status:** Open — Awaiting calibration data from Geopan (Hornresp reference runs)
**Last Updated:** 2026-05-05
**Related:** pyhorn BACKLOG.md (CRIT-1), GdB/sprints/2026-05-05.md

---

## Background

CRIT-1 was the "LF response voltage-independent" bug: pyhorn's low-frequency SPL did not change when driver voltage was varied. The root cause was a mass term (`jωM_rc`) in the rear chamber impedance that Hornresp does NOT use for BLH coupling chambers.

**Partial fix (May 5, 2026):** The `chamber_type: coupling` model was introduced in `RearChamber`, which computes a **pure compliance** (stiffness only) impedance:
```
Z_rc = 1 / (j·ω·C_rc)      where C_rc = V_rc / (ρ·c²)
```
This is the correct physics for a BLH rear chamber — the large sealed box behind the driver acts as an acoustic spring, not a Helmholtz resonator.

**Remaining work:** Geometry-specific calibration of `V_rc` and `l_rc` against Hornresp reference data.

---

## Rear Chamber Models

| `chamber_type` | Physics | When to Use |
|---|---|---|
| `"sealed"` | Pure compliance: `Z = 1/(jωC)` | Standard sealed box rear load |
| `"coupling"` | Pure compliance: `Z = 1/(jωC)` | **BLH rear chamber (correct)** — large sealed box coupling the driver to the horn path |
| `"vented"` | Helmholtz resonator: `Z = 1/(jωC) + jωM` | Bass-reflex box with a port — **not correct for BLH** |

**Important:** `"vented"` was the default before the CRIT-1 fix. It introduces a mass term that produces catastrophically wrong LF response (−13 to −25 dB vs Hornresp at 20–50 Hz). The `"coupling"` model (no mass term) is the correct physics for a BLH coupling chamber.

---

## Physical Meaning of Parameters

### `V_rc` — Rear Chamber Volume (m³)

The acoustic compliance of the rear chamber is:
```
C_rc = V_rc / (ρ · c²)
```

- **Typical values:** 0.005–0.050 m³ (5–50 litres) for a 170 L cabinet
- **Effect on SPL:** Larger V_rc → lower C_rc stiffness → lower system resonance → deeper bass extension
- **Relation to box dimensions:** For a rectangular box, V_rc = width × height × depth
- **Driver's effective VAS contribution:** The rear chamber adds to the driver's suspension compliance; total effective compliance = C_ms + C_rc

### `l_rc` — Rear Chamber Average Length (m)

In a **vented** rear chamber (`chamber_type: "vented"`), `l_rc` is the port length and drives the Helmholtz resonance:
```
f_b = (1 / 2π) · √(A_port / (V_rc · L_eff))
```
where `L_eff ≈ l_rc + 0.6·√(A_port/π)` (end-correction).

In a **coupling** rear chamber (`chamber_type: "coupling"`), `l_rc` is a geometric parameter (average acoustic path length) but does **not** appear in the impedance formula. It is stored for documentation and future multi-parameter fitting.

> **CRIT-1 finding:** For `chamber_type: "coupling"`, `l_rc` does NOT affect the simulation physics. The acoustic model is purely `Z = 1/(jωC)`. This means varying `l_rc` alone in coupling mode should NOT change the SPL — but varying `V_rc` should.

---

## Calibration Approach

### Step 1 — Acquire Hornresp Reference Data

We need Hornresp output for the `hornresp_gdb1` benchmark at **V = 2.83 V** with:
- LF SPL tabulation: 50–200 Hz (1/12-octave or finer)
- Impedance tabulation: same frequency range
- At least two rear chamber configurations (different V_rc) to confirm sensitivity

**Waiting on:** Geopan — email sent 2026-05-05

### Step 2 — Parameter Sweep Design

Once Hornresp data arrives, sweep:

| Parameter | Range | Step | Notes |
|---|---|---|---|
| `V_rc` | 0.005 – 0.050 m³ | 0.005 m³ (5 mL) | 10 values: 5, 10, 15, … 50 L |
| `l_rc` | 0.05 – 0.30 m | 0.05 m | 6 values: 5, 10, 15, 20, 25, 30 cm |

For each combination, compute **LF SPL deviation** vs Hornresp reference:
```
ΔSPL(f) = SPL_pyhorn(f) − SPL_hornresp(f)
```

Target: `|ΔSPL| < 3 dB` for 50–200 Hz across all V_rc values.

### Step 3 — Calibration Metric

Compare at these specific frequencies:
- **System resonance region:** 40–80 Hz (rear chamber compliance dominates)
- **Horn loading region:** 80–200 Hz (horn path dominates)

The **RMS error** across 50–200 Hz:
```
RMS_error = √( Σ [SPL_pyhorn(f) − SPL_hornresp(f)]² / N )
```

---

## Geometry-Specific Nature of Calibration

**This calibration is geometry-specific.** Different cabinets have different rear chamber volumes and geometries, so the optimal `V_rc` and `l_rc` values will differ.

For the **GdB1 BLH**:
- Estimated cabinet volume: ~170 L
- Internal horn displacement: ~15–20 L (estimated from Onshape)
- Available rear chamber volume: ~150 L
- Current calibration: `V_rc = 0.005 m³ (5 L)` ← **This is a TBD value**, not a measurement

For **other projects**, the calibration must be re-done with that project's cabinet geometry.

---

## Current Benchmark: `tests/benchmarks/hornresp_gdb1/`

```yaml
# gdb1_hornresp.yaml — current calibration parameters
rear_chamber:
  vrc: 0.005    # 5 L
  lrc: 0.15     # 15 cm
  chamber_type: coupling
```

The `hornresp_spl.csv` reference file is **STALE** — it was generated with the old `vented` model before CRIT-1 was fixed. It must be regenerated once Hornresp calibration data is received from Geopan.

---

## What to Compare Against Hornresp

When Geopan's reference data arrives, compare:

1. **LF SPL (50–200 Hz)** at `V = 2.83 V`
   - pyhorn: `result.spl` or `result.spl_power_based`
   - Hornresp: `SPL` column from the Hornresp export

2. **System impedance magnitude (20–500 Hz)**
   - pyhorn: `result.impedance` (complex) → `np.abs(result.impedance)`
   - Hornresp: `Zin` column from Hornresp export

3. **Key diagnostic frequencies:**
   - `f_resonance` — impedance peak (driver + rear chamber resonance)
   - `f_cutoff` — horn cutoff frequency (~49 Hz for GdB1 hyperbolic T=0.35)
   - `f_horn_loading` — where horn path begins to dominate (>~80 Hz)

---

## Validation: V_rc Sensitivity Test

Before full Hornresp calibration, a sanity check can confirm the rear chamber IS affecting the simulation:

```python
def test_rear_chamber_calibration_sweep():
    """Confirm V_rc changes SPL — rear chamber IS in the circuit."""
    # Run with V_rc = 0.010, 0.020, 0.030, 0.040, 0.050 m³
    # All other parameters fixed at hornresp_gdb1 benchmark values
    # Assert SPL at 60 Hz differs by > 0.5 dB between runs
```

This test is added to `pyhorn_core/tests/test_solver_models.py` as `test_rear_chamber_calibration_sweep()`.

---

## Open Questions

1. **Geopan reference data:** Awaiting Hornresp runs from Geopan with known V_rc values
2. **`l_rc` role in coupling model:** Does `l_rc` affect any part of the coupling chamber TMM? Currently it does not appear in `rear_chamber_impedance()` for `chamber_type: "coupling"` — verify this is intentional
3. **`fr_tuning` parameter:** In the `RearChamber` dataclass, `fr_tuning` is documented but not used in the coupling model. Should it affect the coupling compliance?
4. **Throat chamber (Vtc/Atc):** Also in the signal chain — should calibration include Vtc sweep too?

---

## References

- Hornresp manual: pages 013 (throat adapter), 048–049 (Compound Horn), 057–058 (Tapped Horn), 065 (Slavbas)
- CRIT-1 Deep Dive: `GdB/sprints/2026-05-05.md`
- Backlog: `pyhorn/BACKLOG.md` (CRIT-1)
- `pyhorn_core/config/models.py` — `RearChamber`, `rear_chamber_impedance()`
- `pyhorn_core/pyhorn_physics/orchestrators.py` — `_horn_response_impl()` rear chamber section
