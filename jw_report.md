# BEMPPSolver vs PyHorns Analysis

## Overview

| Aspect | BEMPPSolver | PyHorns |
|--------|-------------|---------|
| **Method** | Boundary Element Method (BEM) | Transfer Matrix Method (TMM) |
| **Dimensionality** | Full 3D | 1D axial |
| **Input** | Gmsh mesh (.msh) | Parametric YAML |
| **Output** | 3D pressure fields, polar plots | SPL, impedance, group delay, distortion |

---

## What BEMPPSolver Does That PyHorns Doesn't

### 1. Burton-Miller Formulation for Exterior Acoustics
BEMPPSolver uses the Burton-Miller formulation to avoid fictitious interior resonances:
```
lhs = 0.5 * Identity - DLP - coupling * (-hypersingular)
coupling = 1j / k
```
PyHorns uses the standard TMM which doesn't encounter these spurious resonances but also doesn't model full 3D wave interactions.

### 2. Mesh-Based 3D Geometry
BEMPPSolver:
- Loads actual mesh files with physical tags distinguishing throat vs enclosure
- Uses P1 (continuous linear) and DP0 (discontinuous constant) function spaces
- Handles arbitrary 3D geometry via triangle discretization

PyHorns:
- Parametric 1D profiles (exponential, conical, hyperbolic, tractrix)
- 3D effects approximated via radiation impedance formulas

### 3. Spatial Hash + Union-Find Mesh Cleaning
`cleanmesh.py` implements:
- Grid-based spatial hashing for efficient vertex proximity detection
- Union-Find algorithm for transitive vertex merging
- Handles mesh stitching seams from CAD export

### 4. Full Wave Radiation Calculation
BEMPPSolver computes actual acoustic pressure at 3D observation points using single-layer and double-layer potential operators:
- GMRES iterative solver for linear system
- Evaluates pressure field at polar coordinates (horizontal/vertical planes)
- Reference angle normalization with angular wrapping

PyHorns uses analytical piston radiation formulas (Levine/Inglis for circular, Morse & Ingard for rectangular).

### 5. Parallel Frequency Sweep
BEMPPSolver uses `ProcessPoolExecutor` with `spawn` context (critical for macOS) to parallelize independent frequency solutions.

### 6. Acoustic Impedance from Mesh Integration
Calculates impedance by integrating pressure over actual throat surface elements:
```python
Z = F / v  where F = ∫p dS (over throat elements)
```

---

## Recommendations for PyHorns

### High Priority
1. **Add BEM validation mode** - Use BEMPPSolver mesh output to validate TMM predictions
2. **Radiation impedance enhancement** - Incorporate measured/computed 3D radiation patterns
3. **Numerical stability improvements** - Segment merging and kL/segment checks already exist but could be enhanced

### Medium Priority
4. **3D directivity visualization** - Compare polar plots to BEM-computed directivity
5. **Mesh import capability** - Allow importing BEMPPSolver meshes for TMM "calibration"
6. **Parallel TMM sweep** - PyHorns already computes all frequencies in one pass; could parallelize for very complex horns

### Low Priority
7. **Frequency-dependent radiation** - BEMPPSolver shows how directivity varies with frequency in full 3D
8. **Near-field coupling effects** - BEM captures mutual coupling between throat and enclosure

---

## Key Insight

The fundamental trade-off is **accuracy vs speed**:
- **BEM (BEMPPSolver)**: Physically accurate but slow (~minutes per frequency sweep) and requires mesh generation
- **TMM (PyHorns)**: Fast (~seconds) and good for design optimization, but approximates 3D effects

PyHorns could benefit most from:
1. Using BEM results to "calibrate" or validate its radiation impedance models
2. Adding a validation test suite comparing TMM predictions against BEM for standard geometries
