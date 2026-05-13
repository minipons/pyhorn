# Solver Map

## Canonical Code Paths
- Config schemas:
  - `pyhorn_core/config/driver_models.py`
  - `pyhorn_core/config/horn_models.py`
  - `pyhorn_core/config/chamber_models.py`
  - `pyhorn_core/config/project_models.py`
- Config parsing:
  - `pyhorn_core/config/parser.py`
- Physics primitives:
  - `pyhorn_core/pyhorn_physics/radiation.py`
  - `pyhorn_core/pyhorn_physics/__init__.py`
- Calibration and normalization helpers:
  - `pyhorn_core/pyhorn_physics/calibration.py`
- Result container:
  - `pyhorn_core/pyhorn_physics/results.py`
- Public solver facade:
  - `pyhorn_core/pyhorn_physics/orchestrators.py`

## Ownership By Behavior
- Mouth radiation and directivity: `radiation.py`
- Acoustic power normalization and SPL conversions: `calibration.py`
- BLH, TL, TH, CH public entry points: `orchestrators.py`
- Driver, horn, chamber shape: split config modules

## Compatibility Layers
- `pyhorn_core/config/models.py`: backward-compatible re-export facade.
- `pyhorn_core/solver/models.py`: backward-compatible solver facade.

## Refactor Rule
When adding new behavior, prefer creating or extending a focused module instead of growing `orchestrators.py` or `config/models.py`.
