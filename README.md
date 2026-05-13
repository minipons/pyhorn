# pyhorn

Acoustic horn loudspeaker simulator — TMM (Transfer Matrix Method) solver + CLI.

Predicts SPL, impedance, group delay, and diaphragm excursion for back-loaded horn (BLH) and front-loaded horn (FLH) enclosures.

## Quick Start

```bash
pip install -e .
pyhorn calculate -d drivers/FE166NV2.yaml -h projects/bkhiro.yaml --fmin 20 --fmax 2000
```

## CLI Commands

```bash
# Simulate a horn
pyhorn calculate -d drivers/FE166NV2.yaml -h projects/bkhiro.yaml

# Compare multiple horns on one plot
pyhorn compare horns/*.yaml -d drivers/FE166NV2.yaml

# Derive T-S parameters from spec sheet
pyhorn derive-ts --fs 53 --qes 0.29 --qms 5.9 --vas 17.5 --re 8 --sd 124

# Auto-segment a horn from Onshape JSON export
pyhorn auto-segment -i geometry.json -o horn.yaml --n-segments 20
```

## Output

Simulation results (SPL, impedance, group delay, excursion) are written to `outputs/<horn_name>/`:
- `response.csv` — full frequency-by-frequency data
- `response_plot.png` — triple-plot (SPL / impedance / excursion)
- `horn_2d_folded.png` — 2D folded horn schematic
- `horn_3d.png` — 3D unwrapped wireframe
- `csd_waterfall.png` — cumulative spectral decay
- `report.txt` — text summary

## Architecture

- `pyhorn_core/` — TMM solver, physics models, config parsing
- `pyhorn_cli/` — Typer CLI interface
- `drivers/` — driver YAML files (T-S parameters)
- `projects/` — horn project YAML files
- `source/` — horn geometry source files
- `tests/` — benchmarks and test suite

Agent-facing repo maps live in `AGENTS.md` and `docs/architecture/`. Use those before widening changes across solver, benchmark, or config surfaces.

## Packages

| Package                              | Description                                                |
| ------------------------------------ | ---------------------------------------------------------- |
| `pyhorn_core.solver.models`          | Core TMM horn response                                     |
| `pyhorn_core.solver.hornresp_parser` | Hornresp `.txt` file parser                                |
| `pyhorn_core.config.models`          | `DriverSpecs`, `HornGeometry` dataclasses                  |
| `pyhorn_core.pyhorn_physics`         | Acoustic physics (radiation impedance, rear chamber, etc.) |

## Dependencies

- Python ≥ 3.10
- numpy, scipy, matplotlib, pyyaml, typer, shapely, networkx
