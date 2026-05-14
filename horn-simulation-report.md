# horn-simulation vs pyhorn: Implementation Comparison Report

## Overview

| Aspect | horn-simulation | pyhorn |
|--------|-----------------|--------|
| **Method** | FEM (FEniCSx/dolfinx) Helmholtz equation | Transfer Matrix Method (TMM) |
| **Dimensionality** | Full 3D with STEP geometry | 1D cylindrical segment discretization |
| **Radiation model** | BEM (bempp-cl) + analytical piston | Analytical piston (Levine/Inglis, Morse & Ingard, Miki) |
| **Orchestration** | Nextflow pipeline (monorepo) | Standalone Python package |
| **Output** | Self-contained HTML ranking reports | Plots (matplotlib), CSV/FRD/WAV export |

---

## What horn-simulation Does Better

### 1. True Wave-Equation Physics via FEM

**horn-simulation** solves the Helmholtz equation `∇²p + k²p = 0` using finite elements:

```
∫(∇p·∇q − k²pq)dx − jk∫_outlet p·q ds = 0
```

**pyhorn** uses TMM, which assumes plane-wave propagation within each cylindrical segment. This breaks down when:
- The horn cross-section is large relative to wavelength (ka > 1)
- Sharp discontinuities exist
- Higher-order modes develop

**Implication:** horn-simulation captures actual wave phenomena (reflections, diffraction, mode coupling) that TMM cannot model.

### 2. Full 3D Geometry with Adaptive Meshing

**horn-simulation:**
- Generates real STEP files via gmsh + OpenCASCADE
- Lofted circular cross-sections for smooth geometry
- **λ/6 adaptive meshing rule:** `h_adaptive = c₀ / (6 × f_max)` — automatically adjusts mesh density based on frequency
- Boundary tagging by surface center of mass (z-coordinate)

**pyhorn:**
- Profiles discretized into N cylindrical segments
- No actual spatial geometry
- Segment count is a user-chosen parameter, not physics-adaptive

**Implication:** horn-simulation produces geometry that can be 3D-printed; pyhorn cannot.

### 3. Exterior Radiation via BEM

**horn-simulation** offers three radiation models:
- **plane_wave:** ρ₀c₀ (infinite baffle approximation)
- **flanged_piston:** Levine-Schwinger with Struve functions
- **unflanged_piston:** Levine-Schwinger approximation
- **bem:** Nonlocal BEM coupling via bempp-cl (Burton-Miller formulation with GMRES)

The BEM model couples exterior radiation self-consistently with the FEM interior solution — no simplifying assumptions about infinite baffle.

**pyhorn** uses only analytical piston models (Rayleigh, Levine/Inglis, Morse & Ingard), which assume semi-infinite baffle or infinite baffle conditions.

**Implication:** horn-simulation's BEM correctly models the radiation from an unflanged mouth in free space.

### 4. Two-Phase Driver-Horn Coupling

**horn-simulation** decouples FEM simulation from driver parameters:

**Phase A:** FEM with unit pressure at inlet → extracts throat impedance Z_horn (once per geometry)

**Phase B:** Uses Thiele-Small parameters to compute actual SPL:
```
V_g → Z_e → Z_mech_driver → Z_horn → Z_mech_load → Z_mot → diaphragm velocity → throat pressure
```

This means you can evaluate hundreds of driver-horn combinations by re-using the same FEM results.

**pyhorn** couples driver and horn in a single TMM cascade — to evaluate a new driver you must re-run the entire simulation.

### 5. Driver Pre-Screening with EBP-Based Filtering

**horn-simulation** implements intelligent driver pre-screening:
```
1. fs_hz < target_f_low × 1.5         (resonance in/near band)
2. EBP = fs/qes > 50                   (horn suitability)
3. f_piston × horn_load_factor ≥ target_f_high
4. Sd ratio within 0.3–3× of median driver
```

This reduces unnecessary simulations by only evaluating drivers likely to perform well with a horn load.

**pyhorn** has no equivalent pre-screening; it simulates all drivers.

### 6. Automated Geometry Derivation (fullauto mode)

**horn-simulation's geometry_designer.py** analytically derives horn dimensions from target frequency:
```python
r_mouth = c0 / (2π·f_low)  # circumference = wavelength
length_range: quarter-wave to half-wave at f_low
simulation_freq: extended ±0.5 octave for rolloff capture
```

Then generates a **grid of candidates** (profiles × mouth radii × lengths) for systematic exploration.

**pyhorn's optimizer.py** uses differential evolution on a fixed geometry definition — it optimizes parameters within a given design space but cannot derive the initial geometry from a target response.

### 7. Composite Scoring and Ranking Pipeline

**horn-simulation's scoring.py:**
```
bandwidth:  50%
ripple:     25%
sensitivity: 25%
```

The **rank_pipeline.py** produces a self-contained HTML report with:
- Base64-embedded plots
- Driver ranking tables
- T-S parameter tables
- Horn geometry renders

**pyhorn** produces matplotlib plots but has no multi-candidate ranking system.

### 8. Band-Parallelized FEM Pipeline

**horn-simulation's Nextflow pipeline** parallelizes across frequency bands:
- Splits frequency range into N bands
- Each band runs in a separate Docker container
- Results merged via `collect()` and `groupTuple()`

**pyhorn** is single-threaded; parallelization is limited to NumPy vectorized operations.

### 9. Docker Containerization

**horn-simulation** defines 4 Docker containers in `nextflow.config`:
- `horn-geometry` (gmsh, OpenCASCADE)
- `horn-solver` (FEniCSx, PETSc)
- `horn-analysis` (plotting, reporting)
- `horn-bem-solver` (bempp-cl)

Results are **reproducible and cloud-deployable**.

**pyhorn** has no containerization.

### 10. Directivity Computation via BEM

**horn-simulation** computes directivity patterns using Kirchhoff-Helmholtz integral with BEM far-field, producing full 3D directivity plots.

**pyhorn** has a Frequency-Dependent Directivity (FDD) model but no true wave-based directivity calculation.

---

## What pyhorn Does Better

### 1. Broader Enclosure Type Support

pyhorn natively supports:
- Back-Loaded Horn (BLH)
- Front-Loaded Horn (FLH)
- Tapped Horn (TH)
- Compound Horn (CH)
- Finite Transmission Line
- Sealed Box
- Infinite Baffle (bare driver)
- Vented Box (bass reflex)
- Passive Radiator
- Slavic Box (aperiodic)

**horn-simulation** focuses on horn geometry with driver coupling; it has no equivalent for tapped horns, compound chambers, or transmission lines.

### 2. Second Tone Distortion Prediction

pyhorn computes 2nd harmonic distortion analytically from compliance non-linearity (α ≈ 0.3 m⁻¹).

**horn-simulation** does not appear to compute distortion.

### 3. Thermal Power Compression

pyhorn's `compute_thermal_power_compression()` runs a two-pass solver (cold Re vs. hot Re) to compute dB compression from voice coil heating.

**horn-simulation** does not model thermal effects.

### 4. Frequency-Dependent Le Model

pyhorn implements:
- Semi-inductance model for frequency-dependent voice coil inductance
- Lossy Le model with eddy-current loss resistance

**horn-simulation** uses static Thiele-Small parameters without frequency dependence.

### 5. Hornresp Compatibility

pyhorn's `hornresp.py` has a full Hornresp parameter solver (S1, S2, F12, T, Hyp parameters) enabling direct comparison with industry-standard Hornresp software.

**horn-simulation** is not compatible with Hornresp.

### 6. Differential Evolution Optimization

pyhorn's `optimizer.py` uses scipy's differential evolution with a physics-based cost function (flatness + sensitivity + bass extension + excursion penalty + throat area penalty).

**horn-simulation's** fullauto mode generates a grid of candidates but does not use evolutionary optimization.

### 7. Modular CLI with Multiple Output Formats

pyhorn's CLI (`pyhorn_cli/`) supports:
- `calculate`: Full simulation with SPL, impedance, excursion, CSD waterfall, spectrogram
- `compare`: Multi-horn comparison plots
- `derive_ts`: T-S parameter derivation from spec sheets
- Export formats: CSV, JSON, FRD, WAV

**horn-simulation** produces HTML reports and Nextflow log output.

### 8. Folding with NetworkX

pyhorn uses networkx for horn path graph operations when modeling complex folding geometries.

**horn-simulation** generates straight horn paths without folding.

---

## Summary Table

| Feature | horn-simulation | pyhorn |
|---------|-----------------|--------|
| Wave physics fidelity | ✅ FEM (Helmholtz) | ❌ TMM (plane-wave assumption) |
| 3D geometry export | ✅ STEP files | ❌ None |
| Adaptive meshing | ✅ λ/6 rule | ❌ Manual segment count |
| BEM radiation | ✅ bempp-cl | ❌ Analytical only |
| Driver pre-screening | ✅ EBP/f_s/piston filter | ❌ None |
| Transfer function decoupling | ✅ FEM once, couple many drivers | ❌ Must re-run full TMM |
| Automated geometry derivation | ✅ fullauto mode | ❌ Manual specification |
| Composite scoring/ranking | ✅ Yes | ❌ None |
| HTML report generation | ✅ Yes | ❌ Matplotlib only |
| Docker/containerization | ✅ 4 containers | ❌ None |
| Nextflow orchestration | ✅ Yes | ❌ None |
| Enclosure type support | ❌ Horn only | ✅ BLH, FLH, TH, CH, TL, sealed, vented, PR, Slavic |
| Thermal compression | ❌ None | ✅ Two-pass cold/hot |
| Second tone distortion | ❌ None | ✅ Analytical |
| Frequency-dependent Le | ❌ Static T-S | ✅ Semi-inductance + Lossy Le |
| Hornresp compatibility | ❌ None | ✅ Full solver |
| Evolutionary optimization | ❌ Grid search only | ✅ Differential evolution |
| Directivity (wave-based) | ✅ BEM far-field | ❌ FDD model only |

---

## Conclusion

**horn-simulation** represents a more rigorous physics approach (true FEM wave equation, 3D geometry, BEM) and a more production-ready engineering workflow (pre-screening, ranking, containerization, automated reports). Its main limitations are narrow scope (exclusively horn-loading, no tapping/folding/compound chambers) and computational cost (FEM is orders of magnitude slower than TMM).

**pyhorn** is faster, more flexible for enclosure types, and includes psychoacoustic considerations (distortion, thermal compression, frequency-dependent parameters). Its main limitation is the TMM plane-wave assumption, which breaks down for large horns at high frequencies.

**horn-simulation's key advantages pyhorn should consider adopting:**
1. Driver pre-screening based on EBP and piston frequency
2. Transfer function decoupling (FEM once, couple many)
3. Automated initial geometry derivation from target response
4. Composite scoring and multi-candidate ranking with reporting
5. BEM-based exterior radiation model
