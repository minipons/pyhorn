# CRIT-1 Calibration Plan — Geometry-Specific Parameter Extraction

**Status:** Blocked — waiting on Geopan (Hornresp reference data)  
**Owner:** Guillaume de Boyer-Montegut  
**Date:** 2026-05-05  
**Related:** `9f2fc30` (coupling chamber model implemented), sprint `2026-05-05.md`

---

## Background

CRIT-1 is the persistent low-frequency (20–80 Hz) SPL mismatch between pyhorn and Hornresp for the GdB1 (BKHiro) geometry:

| Freq | Hornresp | pyhorn (coupling) | Δraw |
|------|----------|-------------------|------|
| 20 Hz | 73.4 dB | ~50–55 dB | **−18 to −23 dB** |
| 30 Hz | 82.7 dB | ~65–70 dB | **−12 to −17 dB** |
| 40 Hz | 91.0 dB | ~77–82 dB | **−9 to −14 dB** |
| 50 Hz | 100.3 dB | ~100–110 dB | **≈0 dB ✓** |
| 60 Hz | 106.6 dB | ~107–108 dB | **≈0 dB ✓** |
| 80 Hz | 104.7 dB | ~104–105 dB | **≈0 dB ✓** |

The **coupling chamber model** (`chamber_type: coupling`) was implemented in `9f2fc30` and correctly handles the stiffness-dominated behavior of the BLH rear chamber. The remaining gap is **geometry-specific calibration** — the coupling model has a free parameter (`throat_area` of the rear chamber connection) that needs calibration per geometry.

The acoustic impedance of the coupling chamber in pyhorn is:

```
Z_rc = 1 / (jω · C_ab) + Z_rad
C_ab = V_rc / (ρ · c²)
```

Where `C_ab` is the compliance of the rear chamber volume. The acoustic model is correct, but the **absolute calibration** of V_rc and the effective throat connection area (which determines Z_rad loading) must be matched to Hornresp's specific treatment.

---

## What We Need from Geopan

Hornresp uses specific values for the rear chamber that are not fully documented in public Hornresp UI. We need Guillaume to export from Hornresp the following:

### 1. Rear Chamber Parameters (V_rc, L_rc)

For **each geometry** to be calibrated, Hornresp must export the rear chamber tab with:

| Parameter | Description | Units |
|-----------|-------------|-------|
| `Vrc` | Rear chamber volume | litres (L) |
| `Lrc` | Rear chamber effective port length | cm |
| `Fr` or `Fb` | Rear chamber tuning frequency (if vented) | Hz |
| `Drc` or `Diameter` | Port/vent diameter (if vented) | cm |

> ⚠️ **Critical distinction:** For the coupling chamber model, `Lrc` is NOT a vent length — it is the effective length of the throat connection (determines acoustic mass loading at the throat). For the vented model, `Lrc` is the vent length and `Fb` is the Helmholtz tuning frequency.

**Geometries to calibrate:**
1. **BKHiro** (master design) — Vrc=5L, Lrc=15cm (nominal)
2. **BKHiro-resized** (driver-mounted variant)
3. **BK16-MK1**
4. **Hiro** (source geometry)

### 2. Frequency-Point SPL Reference Data

Hornresp must export the full SPL curve (dB/W/m) as CSV for each geometry at:

- **Drive conditions:** Eg = 2.83 V, Rg = 0 Ω (standard sensitivity reference, 1 W into nominal Z)
- **Frequency range:** minimum 10 Hz to 20,000 Hz (logarithmic sweep, ≥ 500 points)
- **Output format:** CSV with columns `Freq (hertz), SPL (dB)`
- **Include:** Full system response (driver + horn + rear chamber + throat chamber)

### 3. Throat Impedance Data (optional but helpful)

Hornresp can export throat impedance (Ztf). This is extremely valuable for calibrating the coupling chamber throat area independently of the SPL response.

### 4. Drive Voltage Calibration

Confirm the drive voltage used in the reference export:
- `Eg = 2.83 V` corresponds to 1 W into `Re = 7.80 Ω` for the FE166NV2
- Confirm whether Hornresp uses `Re` or nominal impedance for the 1 W reference

---

## How to Export from Hornresp

### SPL Export
1. Load the geometry in Hornresp
2. Set drive: `Eg = 2.83 V`, `Rg = 0 Ω`
3. Click **SPL** button (or `Alt+S`)
4. Export: usually `File → Export` or right-click on the SPL graph
5. Save as CSV

### Rear Chamber Parameters
1. Click the **Chamber** tab in Hornresp
2. Screenshot or manually record: `Vrc`, `Lrc`, `Drc` (port diameter), `Fb` (if shown)
3. Note: some parameters may be hidden/collapsed — scroll all panels

### Batch Export (if available)
If Hornresp has a command-line or script export, request a batch export for all geometries:
- `bkhiro.hrd`, `bkhiro_resized.hrd`, `bk16_mk1.hrd`, `hiro.hrd`

---

## What to Do with the Data

Once Geopan provides the data:

### Step 1: Update Reference CSVs
Replace the stale reference CSVs (e.g., `tests/benchmarks/hornresp_gdb1/hornresp_spl.csv`) with exports from the coupling model (same geometry, re-run in Hornresp after confirming the rear chamber is treated as a coupling chamber, not vented).

### Step 2: Calibrate `throat_area` in Coupling Model
The coupling chamber model in pyhorn has a `throat_area` parameter (connection area between rear chamber and horn throat). This is the primary calibration knob. Compare pyhorn's throat impedance vs Hornresp's `Ztf` export to find the correct value.

### Step 3: Validate All Geometries
Run `pytest pyhorn_core/tests/test_benchmark_hornresp.py` after updating reference data. Target: ±3 dB per decade mean delta.

---

## Current Test Status

- `test_benchmark_hornresp.py`: **XFAIL** (4 tests marked xfail) — expected to fail until Geopan data arrives and reference CSVs are regenerated
- `gdb1_hornresp.yaml`: Updated to `chamber_type: coupling` (was `vented`) — fixture now matches the correct model
- `gdb1_horn_only.yaml`: Already has no rear_chamber (HF isolation test — not affected)

---

## Dependencies

- ⏳ **Geopan data** — blocking. Cannot calibrate without Hornresp reference exports.
- `9f2fc30` — coupling chamber model: **DONE**
- Reference CSV regeneration: **BLOCKED on Geopan**
- Per-geometry V_rc / L_rc calibration: **BLOCKED on Geopan**
