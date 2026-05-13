# Hornresp vs pyhorn — Benchmark Comparison

**Status:** Phase 3 comparison tests are blocked, waiting for Geopan to export Hornresp CSV.

This directory contains everything needed to run the comparison once the Hornresp CSV is available.

---

## Quick Start (Guillaume)

```bash
# 1. Generate pyhorn reference output (if needed)
cd /Users/guillaume/P/GdB1
.venv/bin/python -m pyhorn_cli.main calculate calculate \
  --driver tests/benchmarks/hornresp_reference_driver.yaml \
  --horn tests/benchmarks/hornresp_reference_flh.yaml \
  --output-dir tests/benchmarks/output \
  --no-plot --no-plot-3d \
  --fmin 20 --fmax 5000 --n-points 500

# 2. Run comparison (after Geopan provides Hornresp CSV at hornresp/response.csv)
python tests/benchmarks/compare_benchmark.py
```

The comparison script outputs:
- `tests/benchmarks/benchmark_comparison.png` — SPL plot (pyhorn=blue, Hornresp=red, diff=dashed)
- Console summary with RMS error and key metrics

---

## For Geopan — Exporting Hornresp CSV

### Hornresp Parameter Values to Enter

Use these exact values in Hornresp's FLH module:

| Parameter | Value | Notes |
|-----------|-------|-------|
| S1 (throat area) | 40 cm² | |
| S2 (mouth area)  | 300 cm² | |
| L12 (path length) | 152.7 cm | |
| T (hyperbolic constant) | 0.30 | |
| Mmd (moving mass) | 6.12 g | **Important: use 6.12g, not FE166NV2 spec** |
| Ang (radiation angle) | 0.5π sr | Enter as 1.5708 in the angle field |
| Vrc (rear chamber) | 3.24 L | |
| Lrc (rear chamber depth) | 12.0 cm | |
| Vtc (throat chamber) | 88.0 cm³ | |
| Atc (throat chamber area) | 250.18 cm² | |

**Driver parameters (Thiele-Small):**
| Parameter | Value |
|-----------|-------|
| fs | 49.6 Hz |
| Qes | 0.28 |
| Qms | 7.88 |
| Qts | 0.27 |
| Vas | 36.9 L |
| Re | 7.80 Ω |
| Bl | 7.80 N/A |
| Cms | 1.47 mm/N |
| Mmd | 6.12 g |
| Rms | 0.28 kg/s |
| Sd | 132.70 cm² |
| Le | 0.80 mH |

> ⚠️ **Critical:** Use Mmd = 6.12g as shown above — this is the Hornresp reference value,
> NOT the FE166NV2 datasheet value of 6.99g. Using the wrong Mmd will give different results.

---

### How to Export from Hornresp

1. Open Hornresp and enter all parameter values above.
2. Click **Calculate** or run the simulation.
3. When the response graph appears, look for an **Export** or **CSV** button/menu.
   - In many Hornresp versions: **File → Export → CSV** or right-click on the graph.
   - The output should be a CSV with columns: `Frequency (Hz)`, `SPL (dB)`, `Impedance (Ohms)`.
4. Save the file as `response.csv`.
5. Copy it to:
   ```
   /Users/guillaume/P/GdB1/tests/benchmarks/hornresp/response.csv
   ```
   (If you're on Windows and can't access the Mac path directly, let Guillaume know — he can copy it over.)

---

### Verify Your Hornresp Run Before Comparing

Check these expected values from Hornresp:

| Metric | Expected |
|--------|----------|
| Max SPL | **113.4 dB** @ ~1565 Hz |
| SPL at 1 kHz | **107.7 dB** |
| Efficiency peak | **14.1%** @ ~797 Hz |
| Impedance peak | **~26 Ω** @ ~5 kHz |

If your Hornresp values match these (within ±0.2 dB), your run is correct and ready to compare.

---

### Expected pyhorn Output (for cross-check)

pyhorn already produces these values from the reference YAML files:

| Metric | pyhorn |
|--------|--------|
| Max SPL | 113.37 dB @ ~1565 Hz |
| SPL at 1 kHz | **111.79 dB** ⚠️ (+4.09 dB vs Hornresp — this is the known gap) |
| Efficiency peak | 14.13% @ ~797 Hz |
| Impedance peak | 26.02 Ω @ ~5 kHz |

> **Note:** The 4.09 dB discrepancy at 1 kHz is the active issue. The comparison script will show exactly where and by how much the curves diverge.

---

## Directory Structure

```
tests/benchmarks/
├── README_HOW_TO_RUN.md          ← this file
├── compare_benchmark.py          ← comparison script
├── hornresp_reference_driver.yaml ← pyhorn driver input
├── hornresp_reference_flh.yaml   ← pyhorn horn/geometry input
├── output/                       ← pyhorn output (symlink to ../outputs/hornresp_flh_reference/)
│   └── response.csv
└── hornresp/                    ← Geopan's Hornresp export goes here
    └── response.csv             ← ★ TO BE PROVIDED BY GEOPAN ★
```

---

## Running the Comparison

```bash
cd /Users/guillaume/P/GdB1
python tests/benchmarks/compare_benchmark.py
```

Output:
- `tests/benchmarks/benchmark_comparison.png` — visual comparison
- Console: RMS error, max error, key frequency metrics
