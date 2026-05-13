#!/usr/bin/env python3
"""
Hornresp vs pyhorn benchmark comparison script.

Compares pyhorn's response.csv (from hornresp_reference YAML run) against
Hornresp-exported CSV (to be provided by Geopan at tests/benchmarks/hornresp/response.csv).

Usage:
    python tests/benchmarks/compare_benchmark.py
"""

import sys
from pathlib import Path

import numpy as np

# matplotlib settings: use Agg (non-interactive) backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
PYHORN_CSV = SCRIPT_DIR / "output" / "response.csv"
HORNRESP_CSV = SCRIPT_DIR / "hornresp" / "response.csv"
PLOT_OUT = SCRIPT_DIR / "benchmark_comparison.png"

# ── Expected pyhorn column names ─────────────────────────────────────────────
# pyhorn output CSV has this header:
#   Frequency_Hz, SPL_dB_Horn SPL (dB), SPL_dB_Impedance (Ohms), ...
PYHORN_FREQ_COL = "Frequency_Hz"
PYHORN_SPL_COL = "SPL_dB_Horn SPL (dB)"


def load_pyhorn(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load pyhorn response.csv. Returns (frequency, SPL) arrays."""
    import csv

    freq, spl = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            freq.append(float(row[PYHORN_FREQ_COL]))
            spl.append(float(row[PYHORN_SPL_COL]))
    return np.array(freq), np.array(spl)


def load_hornresp(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load Hornresp-exported CSV.
    
    Hornresp export format typically has columns:
      Frequency (Hz), SPL (dB), Impedance (Ohms), ...
    or:
      Freq (Hz), SPL (dB ref 2.83V), ...

    Tries common column name patterns.
    """
    import csv

    freq, spl = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        freq_col = _find_col(headers, ["frequency", "freq", "frequency (hz)", "freq (hz)"])
        spl_col = _find_col(
            headers,
            ["spl", "spl (db)", "spl (db ref 2.83v)", "spl (db ref 2.83v/m)"],
        )
        if freq_col is None or spl_col is None:
            raise ValueError(
                f"Could not identify Frequency/SPL columns in Hornresp CSV.\n"
                f"Available columns: {headers}"
            )
        for row in reader:
            try:
                freq.append(float(row[freq_col]))
                spl.append(float(row[spl_col]))
            except ValueError:
                continue  # skip header-like rows
    return np.array(freq), np.array(spl)


def _find_col(headers: list[str], candidates: list[str]) -> str | None:
    """Case-insensitive column lookup."""
    h_lower = {h.lower().strip(): h for h in headers}
    for c in candidates:
        if c in h_lower:
            return h_lower[c]
    return None


def resample(
    freq_src: np.ndarray,
    spl_src: np.ndarray,
    freq_dst: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate SPL onto destination frequency grid."""
    f_interp = interp1d(
        freq_src,
        spl_src,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    return f_interp(freq_dst)


def rms_error(a: np.ndarray, b: np.ndarray) -> float:
    """RMS difference between two arrays."""
    return float(np.sqrt(np.mean((a - b) ** 2)))


def max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)))


def main() -> int:
    # ── Load pyhorn (always present) ─────────────────────────────────────────
    if not PYHORN_CSV.exists():
        print(f"ERROR: pyhorn output not found at {PYHORN_CSV}", file=sys.stderr)
        print(
            "  Run: pyhorn calculate calculate "
            "--driver tests/benchmarks/hornresp_reference_driver.yaml "
            "--horn tests/benchmarks/hornresp_reference_flh.yaml "
            "--output-dir tests/benchmarks/output --no-plot",
            file=sys.stderr,
        )
        return 1

    pyhorn_freq, pyhorn_spl = load_pyhorn(PYHORN_CSV)
    print(f"[pyhorn] Loaded {len(pyhorn_freq)} points: {pyhorn_freq.min():.1f}–{pyhorn_freq.max():.1f} Hz")

    # ── Load Hornresp (wait for Geopan) ──────────────────────────────────────
    if not HORNRESP_CSV.exists():
        print(
            f"\nWARNING: Hornresp CSV not found at {HORNRESP_CSV}",
            file=sys.stderr,
        )
        print(
            "  Place Geopan's Hornresp export at: tests/benchmarks/hornresp/response.csv",
            file=sys.stderr,
        )
        print("  Then re-run: python tests/benchmarks/compare_benchmark.py\n", file=sys.stderr)
        # Still plot pyhorn alone so something is visible
        hornresp_freq = None
        hornresp_spl = None
    else:
        hornresp_freq, hornresp_spl = load_hornresp(HORNRESP_CSV)
        print(
            f"[Hornresp] Loaded {len(hornresp_freq)} points: "
            f"{hornresp_freq.min():.1f}–{hornresp_freq.max():.1f} Hz"
        )

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(pyhorn_freq, pyhorn_spl, "b-", linewidth=1.2, label="pyhorn", alpha=0.85)

    if hornresp_freq is not None:
        # Resample Hornresp onto pyhorn frequency grid for comparison
        hornresp_spl_rs = resample(hornresp_freq, hornresp_spl, pyhorn_freq)
        ax.plot(hornresp_freq, hornresp_spl, "r-", linewidth=1.2, label="Hornresp", alpha=0.85)

        # ── Stats ────────────────────────────────────────────────────────────
        diff = pyhorn_spl - hornresp_spl_rs

        print("\n" + "=" * 60)
        print("BENCHMARK COMPARISON SUMMARY")
        print("=" * 60)
        print(f"  RMS error:          {rms_error(pyhorn_spl, hornresp_spl_rs):.3f} dB")
        print(f"  Max absolute error: {max_abs_error(pyhorn_spl, hornresp_spl_rs):.3f} dB")
        print(f"  Mean error:         {np.mean(diff):+.3f} dB")
        print(f"  Std of error:       {np.std(diff):.3f} dB")

        # Key frequency metrics
        idx_1khz = np.argmin(np.abs(pyhorn_freq - 1000))
        idx_max = np.argmax(pyhorn_spl)
        print(f"\n  Key values (pyhorn):")
        print(f"    Max SPL:     {pyhorn_spl[idx_max]:.2f} dB @ {pyhorn_freq[idx_max]:.0f} Hz")
        print(f"    SPL @ 1 kHz: {pyhorn_spl[idx_1khz]:.2f} dB")

        if hornresp_freq is not None:
            idx_1khz_h = np.argmin(np.abs(hornresp_freq - 1000))
            idx_max_h = np.argmax(hornresp_spl)
            print(f"\n  Key values (Hornresp):")
            print(f"    Max SPL:     {hornresp_spl[idx_max_h]:.2f} dB @ {hornresp_freq[idx_max_h]:.0f} Hz")
            print(f"    SPL @ 1 kHz: {hornresp_spl[idx_1khz_h]:.2f} dB")

        print("=" * 60 + "\n")

        # Difference curve (dashed)
        ax.plot(
            pyhorn_freq,
            diff,
            "k--",
            linewidth=0.8,
            label=f"pyhorn − Hornresp (RMS={rms_error(pyhorn_spl, hornresp_spl_rs):.2f} dB)",
            alpha=0.6,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("SPL (dB)")
    ax.set_title("Hornresp vs pyhorn — FLH Benchmark SPL Comparison")
    ax.legend(loc="best")
    ax.grid(which="both", linewidth=0.3, alpha=0.4)
    ax.set_xlim(20, 5000)
    ax.set_ylim(50, 120)

    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    print(f"Plot saved → {PLOT_OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
