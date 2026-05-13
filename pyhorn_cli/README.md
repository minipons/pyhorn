# pyhorn_cli — Acoustic Horn Simulator CLI

CLI wrapper for `pyhorn_core` — Typer-based command-line interface for simulation, comparison and synthesis management.

## Installation

```bash
pip install -e .
pyhorn --help          # or: python -m pyhorn_cli.main --help
```

## Top-Level Commands

pyhorn exposes four top-level commands:

| Command                 | Description                                                      |
| ----------------------- | ---------------------------------------------------------------- |
| `pyhorn calculate`      | Horn simulation commands (subcommands below)                     |
| `pyhorn segment-wizard` | Geometry calculator for a single catenoidal horn segment         |
| `pyhorn chamber-wizard` | Estimate Vrc, Lrc, Vtc, Atc, Ap1, Lpt from driver T-S parameters |
| `pyhorn resize-wizard`  | Scale a horn geometry proportionally (Hornresp page 68)          |

---

## `pyhorn calculate` — Simulation Subcommands

All simulation commands live under `pyhorn calculate`:

### `pyhorn calculate calculate`
Simulate the acoustic response of a horn enclosure and save results.

```bash
pyhorn calculate calculate -d drivers/FE166NV2.yaml -p projects/hirob.yaml
pyhorn calculate calculate -d drivers/FE166NV2.yaml -h tests/benchmarks/hornresp/hirob/fixture/horn.yaml
```

Key options:
- `-d/--driver` — driver YAML/JSON config (required)
- `-p/--project` — project YAML (carries geometry path + driver coords + metadata)
- `-h/--horn` — direct geometry YAML (alternative to `--project`)
- `-o/--output-dir` — output directory (default: `outputs`)
- `--fmin/--fmax` — frequency range in Hz (default: 20–5000 Hz)
- `--n-points` — number of frequency points (default: 500)
- `--export-csv / --no-export-csv` — export CSV (default: enabled)
- `--plot / --no-plot` — generate response plot PNG (default: enabled)
- `--export-json` — export JSON (default: disabled)
- `--filter` — apply Filter Wizard from YAML config
- `--polar-freq` — polar directivity frequency (Hz)
- `--polar-angles` — comma-separated off-axis angles (e.g. `0,15,30,45,60,75,90`)
- `--notch-filter` — suppress TMM artifact notches
- `--distortion` — compute 2nd-tone distortion (single-segment horns)
- `--voice-coil-temp` — voice coil temperature for thermal model
- `--fdd` / `--fdd-fc` / `--fdd-dmax` — frequency-dependent directivity mode
- `--benchmark` — run against `tests/benchmarks/hornresp/hirob/fixture/horn.yaml`

### `pyhorn calculate compare`
Compare SPL responses of multiple horn designs on a single plot.

```bash
pyhorn calculate compare projects/hirob.yaml tests/benchmarks/hornresp/hirob/fixture/horn.yaml -d drivers/FE166NV2.yaml
```

Key options:
- `horns...` — list of horn YAML configs (required)
- `-d/--driver` — driver YAML (required)
- `-o/--output-dir` — output directory (default: `outputs/comparison`)
- `--target-spl` — reference SPL line

### `pyhorn calculate derive-ts`
Derive Thiele-Small SI parameters from spec-sheet measurements.

```bash
pyhorn calculate derive-ts --fs 53 --qes 0.29 --qms 5.9 --vas 17.5 --re 8 --sd 124
```

Outputs values in SI units ready for use in pyhorn configurations.

### `pyhorn calculate hornresp`
Solve Hornresp S1/S2/F12/T/Hyp inputs and emit a pyhorn horn definition.

```bash
pyhorn calculate hornresp --s1 40 --s2 300 --f12 50 --t 0.3 --output horn.yaml
```

Key options:
- `--s1/--s2/--f12/--t/--hyp` — Hornresp geometry parameters
- `--enclosure` — FLH or BLH (default: BLH)
- `--n-segments` — discretisation segments (default: 100)
- `--lrc/--vrc/--vtc` — chamber parameters

### `pyhorn calculate tapped-horn`
Simulate a Tapped Horn (TH / TH1 mode). The driver is positioned at an interior point of the horn.

```bash
pyhorn calculate tapped-horn -d drivers/FE166NV2.yaml --th-config tapped.yaml
```

### `pyhorn calculate throat-adapter`
Compute a throat adapter geometry and emit a YAML snippet ready for your project.

```bash
pyhorn calculate throat-adapter --d1 50 --d2 100 --a1 30 --a2 30 --type conical
```

Computes minimum-length profile between throat chamber opening and horn throat. Prints `ap1`, `lpt`, `type` parameters for pasting into project YAML.

### `pyhorn calculate auto-segment`
Generate pyhorn segments automatically from an Onshape 2D air volume JSON export.

```bash
pyhorn calculate auto-segment -i path/to/onshape-export.json -o hirob_imported.yaml --n-segments 20
```

Key options:
- `-i/--input` — Onshape JSON export path
- `-o/--output` — output YAML path (required)
- `-c/--from-clipboard` — read JSON from clipboard (macOS)
- `--n-segments` — number of conical segments
- `--throat-adapter-d1/--throat-adapter-d2` — embed throat adapter geometry
- `--profile-type` — profile for each segment (exponential, catenoidal, hyperbolic, conical)
- `--output-format` — `segments` (default, emits `sections:`) or `rectangular`

### `pyhorn calculate diagnose-spl`
Diagnose SPL response for artifacts and standing-wave patterns.

```bash
pyhorn calculate diagnose-spl -d drivers/FE166NV2.yaml -h examples/geometry/hirob.yaml
```

Analyses a frequency sub-range for: smoothness score, standing-wave analysis (path-length comb filtering), and artifact flagging.

### `pyhorn calculate optimize`
Find optimal horn geometry for a given driver using scipy differential_evolution.

```bash
pyhorn calculate optimize -d drivers/FE166NV2.yaml --fmin 80 --enclosure BLH
```

Key options:
- `-d/--driver` — driver YAML (required)
- `--fmin/--fmax` — target band frequency range (default: 80–5000 Hz)
- `--enclosure` — FLH or BLH (default: BLH)
- `--max-path-length/--max-mouth-area` — physical constraints
- `--profiles` — comma-separated profile types (default: all four)
- `--max-iter` — max iterations per profile (default: 150)
- `--top-n` — number of top designs to output (default: 3)

## Wizard Commands (Top-Level Shortcuts)

The wizard commands are also available as top-level shortcuts:

```bash
pyhorn segment-wizard --s1 40 --s2 300 --l12 1.5 --f12 50
pyhorn chamber-wizard
pyhorn resize-wizard
```

These match `pyhorn calculate segment-wizard`, `pyhorn calculate chamber-wizard`, and `pyhorn calculate resize-wizard` respectively.

---

## Examples

```bash
# Simulate from a project file
pyhorn calculate calculate -d drivers/FE166NV2.yaml -p projects/hirob.yaml

# Compare multiple horns
pyhorn calculate compare projects/hirob.yaml tests/benchmarks/hornresp/hirob/fixture/horn.yaml -d drivers/FE166NV2.yaml

# Derive T-S parameters
pyhorn calculate derive-ts --fs 53 --qes 0.29 --qms 5.9 --vas 17.5 --re 8 --sd 124

# Auto-segment from Onshape export
pyhorn calculate auto-segment -i path/to/onshape-export.json -o hirob_imported.yaml --n-segments 20

# Compute throat adapter
pyhorn calculate throat-adapter --d1 50 --d2 100 --a1 30 --a2 30 --type conical

# Diagnose SPL artifacts
pyhorn calculate diagnose-spl -d drivers/FE166NV2.yaml -h examples/geometry/hirob.yaml

# Optimize geometry for a driver
pyhorn calculate optimize -d drivers/FE166NV2.yaml --fmin 80 --enclosure BLH

```

## Dependencies

pyhorn_core, typer >= 0.9.0
