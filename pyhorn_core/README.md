# pyhorn_core — Acoustic Horn Simulator Core Library

Headless acoustic horn loudspeaker simulator — core physics and numerics only, no CLI or UI dependencies.

## Quickstart

```python
from pyhorn_core import parse_driver_specs, parse_horn_geometry, horn_response

driver = parse_driver_specs("fostex.yaml")       # Thiele-Small parameters
horn   = parse_horn_geometry("bkhiro.yaml")       # horn path & cross-sections
result = horn_response(driver, horn, fmin=20, fmax=20000)
print(result.spl_db)  # array of SPL values (dB)
```

## Installing

`pyhorn_core` is installed automatically with the full `pyhorn` package:

```bash
pip install pyhorn          # core simulator only
pip install pyhorn[dev]     # + test dependencies (pytest, pytest-cov)
```

For development, clone the repo and install in editable mode:

```bash
git clone https://github.com/gdebyser/pyhorn.git
cd pyhorn
pip install -e ".[dev]"
```

## Design Wizards

The pyhorn toolchain includes four interactive design wizards built on top of `pyhorn_core`:

| Wizard               | One-line description                                                                                                   |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **chamber-wizard**   | Computes sealed / vented / passive-radiator rear chamber volume from Thiele-Small parameters and a target -3 dB cutoff |
| **segment-wizard**   | Splits a horn path into manufacturable straight-axis segments (staircase approximation) with area constraints          |
| **synthesis-wizard** | Synthesises a target horn geometry from a desired cardioid orond directivity pattern                                   |
| **hornresp**         | Imports and exports Hornresp-format XML files, bridging legacy `.hornresp` projects with pyhorn YAML                   |

## Architecture

```
pyhorn_cli  /  pyhorn_api  /  pyhorn_ui
                    │
               pyhorn_core
         ┌─────┬─────┼──────────────────┐
         │     │     │                  │
    ┌────┴─┐┌─┴──┐┌─┴───┐   ┌───────────────┐
    │physics│ │fold││segment│   │   wavefront  │
    │(TMM)  │ │SLSQP││medial│   │   (Helmholtz)│
    └────┬─┘ └────┘└─────┘   └───────────────┘
         │       │              │
    ┌────┴───────┴──────┬──────┴────────┐
    │  pyhorn_core/solver/               │
    │  profiles · hornresp · geometry    │
    │  discretise · scoring · time_domain │
    │  chamber_wizard · synthesis_wizard  │
    └──────────────────────────────────┘
                            │
                     ┌──────┴──────┐
                     │  registry   │
                     │  (flat-file)│
                     └─────────────┘
```

**Legend**

| Box        | Location                      | Notes                                                   |
| ---------- | ----------------------------- | ------------------------------------------------------- |
| `physics`  | `pyhorn_core/pyhorn_physics/` | TMM solver, Miki (1990) absorption, acoustic primitives |
| `segment`  | `pyhorn_segment` (extracted)  | Medial-axis computation and auto-segment pipeline       |
| `registry` | `pyhorn_registry` (extracted) | Flat-file driver/project index at `~/.pyhorn/`          |
| `solver/`  | `pyhorn_core/solver/`         | Profile functions, Hornresp parser, geometry, scoring   |

`pyhorn_core` owns the `physics` engine (TMM with Miki 1990 frequency-dependent absorption) and the surrounding numerics (`solver/`). The supported extracted sibling packages are `pyhorn_segment` and `pyhorn_registry`.

## Extracted Packages

| Package           | One-line description                                                                                                       | Hornresp manual |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------- |
| `pyhorn_registry` | Flat-file driver & project registry at `~/.pyhorn/` — YAML files indexed by `registry.json` for fast name-based lookup     | —               |
| `pyhorn_segment`  | Medial-axis computation and auto-segment pipeline — generates manufacturable `sections` from Onshape JSON geometry exports | §Segmentation   |

These packages live in the same repository at `/Users/guillaume/P/GdB1/` and are installed as regular dependencies of `pyhorn_core` when needed.

> **Note on shims:** `pyhorn_core/registry.py` and `pyhorn_core/solver/models.py` are backward-compat re-export shims. All live code for the registry lives in `pyhorn_registry`; all physics orchestrators (`horn_response`, `SimulationResult`, etc.) live in `pyhorn_physics.orchestrators`. Update your imports if you reference these internal paths.

**Headless** means it has no GUI and no command-line interface. It is a library you import from Python scripts or a REST API (e.g. [pyhorn_ui](https://github.com/gdebyser/pyhorn_ui)).

## What it does

Computes SPL, electrical impedance, excursion, group delay, and phase for front-loaded (FLH) and back-loaded (BLH) horns using transfer-matrix methods (TMM) with Miki (1990) frequency-dependent absorption.

## Packages

| Package                      | Contents                                                                                                                                                                                    |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `pyhorn_core.solver`         | Profile functions, Hornresp parser, geometry discretisation, scoring, time-domain; re-export shim for physics orchestrators                                                                 |
| `pyhorn_core.pyhorn_physics` | Physics engine: TMM solver orchestrators (`horn_response`, `horn_response_tapped`, `horn_response_compound`), acoustic primitives (radiation impedance, Miki absorption, transfer matrices) |
| `pyhorn_core.config`         | `DriverSpecs` / `HornGeometry` dataclasses, YAML/JSON parser                                                                                                                                |
| `pyhorn_core.output`         | CSV/JSON exporter, matplotlib plotting                                                                                                                                                      |
| `pyhorn_core.registry`       | **Re-export shim** — actual code lives in `pyhorn_registry`                                                                                                                                 |

## Quick start (full example)

```python
from pyhorn_core import parse_driver_specs, parse_horn_geometry, horn_response

driver = parse_driver_specs("drivers/FE166NV2.yaml")
horn   = parse_horn_geometry("examples/geometry/hirob.yaml")
result = horn_response(driver, horn, fmin=20, fmax=20000)

print(result.spl_db)       # dB SPL array
print(result.z_real)       # electrical resistance (ohms)
print(result.excursion_mm) # driver excursion (mm)
```

## Main functions

### `parse_driver_specs(filepath)`

Parses a driver YAML/JSON file into a `DriverSpecs` dataclass (Thiele-Small parameters). Accepts `.yaml` and `.json` files.

```python
from pyhorn_core import parse_driver_specs

driver = parse_driver_specs("path/to/FE166NV2.yaml")
# driver.fs   → resonant frequency (Hz)
# driver.qts  → total Q
# driver.vas  → equivalent compliance volume (m³)
# driver.sd   → piston area (m²)
# ... and more (see DriverSpecs in config/models.py)
```

### `parse_horn_geometry(filepath)`

Parses a horn geometry YAML/JSON file into a `HornGeometry` dataclass. Handles both legacy single-profile format and the new chained `sections` format.

```python
from pyhorn_core import parse_horn_geometry

horn = parse_horn_geometry("path/to/bk16.yaml")
# horn.throat_area  → initial throat area (m²)
# horn.mouth_area   → final mouth area (m²)
# horn.path_length  → total path length (m)
# horn.sections     → List[Section] (if using sections format)
```

### `parse_horn_project(filepath)`

Parses a project YAML which references a separate geometry file. Applies project-level overrides (driver coordinate, fold plot segments, rear chamber, throat chamber, vented box, passive radiator). Available from:

```python
from pyhorn_core import parse_horn_project              # ✓ recommended
from pyhorn_core.config.parser import parse_horn_project  # same thing
```

### `horn_response(freqs, driver, horn, ...)`

Runs the TMM simulation and returns a `SimulationResult`. Available from three import paths (all resolve to the same function):

```python
from pyhorn_core import horn_response              # ✓ recommended
from pyhorn_core.solver.models import horn_response  # backward-compat re-export shim
from pyhorn_core.pyhorn_physics.orchestrators import horn_response  # canonical location
```

See `pyhorn_core/pyhorn_physics/orchestrators.py` for the full signature.

## Chained profile sections

For complex folded horns, the `sections` field chains multiple profile segments together, each with its own profile type, length, and area transition:

```yaml
sections:
  - name: segment_1
    profile_type: exponential
    start_area: 0.00947   # m²
    end_area: 0.0142      # m²
    length: 0.492         # m
  - name: segment_2
    profile_type: exponential
    start_area: 0.0142
    end_area: 0.0224
    length: 0.146
  - name: segment_3
    profile_type: straight   # constant-area passage
    start_area: 0.0224
    end_area: 0.0224
    length: 0.072
```

Each `Section` accepts:

| Field          | Type    | Description                                                                   |
| -------------- | ------- | ----------------------------------------------------------------------------- |
| `name`         | `str`   | Segment label                                                                 |
| `profile_type` | `str`   | `exponential`, `straight`, `conical`, `hyperbolic`, `catenoidal`, `parabolic` |
| `start_area`   | `float` | Cross-sectional area at the segment start (m²)                                |
| `end_area`     | `float` | Cross-sectional area at the segment end (m²)                                  |
| `length`       | `float` | Segment path length (m)                                                       |
| `hyperbolic_t` | `float` | Miki hyperbolic parameter (only for `hyperbolic` profile type)                |

**Loading a horn with sections:**

```python
from pyhorn_core import parse_horn_geometry

horn = parse_horn_geometry("examples/geometry/hirob.yaml")
# horn.sections → List[Section]
```

`HornGeometry.sections` is the preferred format for new designs. It supports all standard profile functions and geometry-aware discretisation.

## Dependencies

numpy, scipy, matplotlib, pyyaml
