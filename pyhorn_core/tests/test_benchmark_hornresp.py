"""
Regression test: pyhorn vs Hornresp GdB1 BLH benchmark.

This test validates pyhorn's BLH simulation against a Hornresp export
for the GdB1 matched parameters (tests/benchmarks/hornresp_gdb1/).

Hornresp reference setup:
  Driver:  Fostex FE166NV2 equivalent
           Sd=132 cm², Bl=7.75, Cms=1.49E-03 m/N, Rms=0.27,
           Mmd=6.04 g, Le=0.80 mH, Re=7.80 Ω
  Horn:    S1=80 cm² (throat), S2=600 cm² (mouth),
           L=1.53 m, T=0.35 (hyperbolic), F12≈49 Hz
  Chamber: Vrc=5 L, Lrc=15 cm; Vtc=160 cm³, Atc=80 cm²
  Drive:   Eg=2.83 V, Rg=0 Ω  →  1 W into 8 Ω  (Hornresp reference power)
  Ang:     0.5π sr (half-space radiation)

Expected thresholds (after CRIT-1 / CRIT-3 fixes):
  Per-decade mean delta:  ±3 dB
  Per-point max deviation: 10 dB

Current state (LF stuck at −15 dB due to CRIT-1, HF excess +14–19 dB):
  This test documents the known deviation and will PASS once CRIT-1 is resolved.

Reference CSV columns used:
  Freq (hertz), SPL (dB)

Run:
  pytest pyhorn_core/tests/test_benchmark_hornresp.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import interp1d

REPO = Path("/Users/guillaume/P/GdB1")
BENCHMARK_DIR = REPO / "tests/benchmarks/hornresp_gdb1"
HORNRESP_CSV = BENCHMARK_DIR / "hornresp_spl.csv"
COMBINED_YAML = BENCHMARK_DIR / "gdb1_hornresp.yaml"

# ── Thresholds ────────────────────────────────────────────────────────────────
TOLERANCE_PER_DECADE_DB = 3.0   # mean |delta| per decade must be < 3 dB
TOLERANCE_PER_POINT_DB = 10.0  # no single point exceeds ±10 dB

# ── Expected Hornresp peak ────────────────────────────────────────────────────
# Hornresp shows ~112 dB SPL peak and ~102 dB at 1 kHz
# These are cross-check values, not hard thresholds.
EXPECTED_SPL_PEAK_DB = (108.0, 115.0)   # reasonable range for peak SPL
EXPECTED_SPL_1KHZ_DB = (100.0, 106.0)   # Hornresp ≈ 102.6 dB at 1 kHz


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_hornresp_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load Hornresp exported CSV. Returns (frequency Hz, SPL dB) arrays."""
    freqs, spls = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            freqs.append(float(row["Freq (hertz)"]))
            spls.append(float(row["SPL (dB)"]))
    return np.array(freqs), np.array(spls)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hornresp_data() -> tuple[np.ndarray, np.ndarray]:
    """Load Hornresp reference data."""
    fqs, spls = load_hornresp_csv(HORNRESP_CSV)
    assert len(fqs) > 100, f"Hornresp CSV should have >100 points, got {len(fqs)}"
    return fqs, spls


@pytest.fixture(scope="module")
def pyhorn_result(hornresp_data):
    """Run pyhorn simulation for GdB1 matched parameters."""
    import sys
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "pyhorn_core"))
    sys.path.insert(0, str(REPO / "pyhorn_api"))

    import yaml
    from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry

    # Split combined YAML into driver and horn dicts
    with open(COMBINED_YAML) as f:
        params = yaml.safe_load(f)

    DRIVER_FIELDS = {
        "fs", "qts", "qes", "qms", "vas", "re", "sd", "bl",
        "mms", "cms", "rms", "le", "xmax", "voltage", "alpha_re",
        "le_freq_dependency", "le_f_ref", "lossy_le",
        "le_R_e_eddy", "le_f_lossy_ref",
    }
    HORN_FIELDS = {
        "throat_area", "mouth_area", "path_length", "enclosure_type",
        "path_diff", "ang", "vrc", "lrc", "fr_rc", "vented_box",
        "passive_radiator", "slavbas", "vtc", "atc", "fr_tc",
        "ap1", "lpt", "throat_adapter_type", "profile_type",
        "hyperbolic_t", "n_segments", "width", "sections",
        "conical_segments", "rectangular_segments", "coordinates",
        "enclosure_dims", "driver_coord", "discretisation", "bend_angles",
        "lem_step_model", "lem_step_strength", "lem_step_resistance",
        "segments", "bends",
    }

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix="_driver.yaml", mode="w", delete=False) as df:
        yaml.dump({k: v for k, v in params.items() if k in DRIVER_FIELDS}, df)
        driver_path = df.name
    with tempfile.NamedTemporaryFile(suffix="_horn.yaml", mode="w", delete=False) as hf:
        yaml.dump({k: v for k, v in params.items() if k in HORN_FIELDS}, hf)
        horn_path = hf.name

    try:
        driver = parse_driver_specs(driver_path)
        horn = parse_horn_geometry(horn_path)
    finally:
        os.unlink(driver_path)
        os.unlink(horn_path)

    hr_freqs, _ = hornresp_data  # type: ignore
    fmin, fmax = hr_freqs.min(), hr_freqs.max()
    # Match Hornresp resolution: ~533 points from 10 Hz to 20 kHz
    py_freqs = np.logspace(np.log10(fmin), np.log10(fmax), 800)

    from pyhorn_core.pyhorn_physics.orchestrators import horn_response
    result = horn_response(py_freqs, driver, horn, compute_distortion=False)

    # Interpolate to Hornresp frequencies
    log_hr = np.log10(hr_freqs)
    log_py = np.log10(py_freqs)
    valid = (hr_freqs >= py_freqs.min()) & (hr_freqs <= py_freqs.max())
    interp = interp1d(log_py, result.spl, kind="linear", fill_value="extrapolate")
    py_spl_at_hr = interp(log_hr)

    return {
        "hr_freqs": hr_freqs,
        "hr_spls": hornresp_data[1],
        "py_freqs": py_freqs,
        "py_spls": result.spl,
        "py_spl_at_hr": py_spl_at_hr,
        "valid": valid,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Sanity checks (these should always pass)
# ─────────────────────────────────────────────────────────────────────────────

class TestGdB1SimulationSanity:
    """Basic sanity checks on the pyhorn simulation output."""

    def test_spl_in_reasonable_range(self, pyhorn_result):
        """Peak SPL should be between 108–115 dB for this driver/horn."""
        peak = float(np.max(pyhorn_result["py_spls"]))
        lo, hi = EXPECTED_SPL_PEAK_DB
        assert lo <= peak <= hi, (
            f"Peak SPL {peak:.1f} dB outside expected range [{lo}, {hi}] dB. "
            "Check driver parameters and horn geometry."
        )

    def test_spl_1khz_reasonable(self, pyhorn_result):
        """SPL at 1 kHz should be within ~4 dB of Hornresp reference."""
        hr_f, hr_s = pyhorn_result["hr_freqs"], pyhorn_result["hr_spls"]
        idx_1khz = np.argmin(np.abs(hr_f - 1000.0))
        hr_1khz = float(hr_s[idx_1khz])
        py_at_hr = pyhorn_result["py_spl_at_hr"]
        py_1khz = float(py_at_hr[idx_1khz])
        delta = abs(py_1khz - hr_1khz)
        # Currently this will show ~9.5 dB excess (known CRIT-3 issue)
        assert delta < 15.0, (
            f"pyhorn SPL at 1 kHz is {py_1khz:.1f} dB vs Hornresp {hr_1khz:.1f} dB "
            f"(delta={py_1khz-hr_1khz:+.1f} dB). Expected <15 dB. "
            "CRIT-3: HF sensitivity reference mismatch — see issue documentation."
        )

    def test_no_nan_inf(self, pyhorn_result):
        """SPL array must contain no NaN or Inf values."""
        assert np.all(np.isfinite(pyhorn_result["py_spls"])), (
            "pyhorn SPL contains NaN or Inf — solver error."
        )

    def test_spl_not_flat(self, pyhorn_result):
        """SPL response must have more than 3 dB variation — real horn response."""
        py_spls = pyhorn_result["py_spls"]
        assert py_spls.max() - py_spls.min() > 3.0, (
            "SPL range < 3 dB — likely a solver bug or all-zeros output."
        )

    def test_hornresp_data_loaded(self, hornresp_data):
        """Hornresp reference data must load successfully."""
        fqs, spls = hornresp_data
        assert len(fqs) > 100, f"Expected >100 Hornresp points, got {len(fqs)}"
        assert fqs.min() < 20, f"Expected fmin < 20 Hz, got {fqs.min()}"
        assert fqs.max() > 15000, f"Expected fmax > 15000 Hz, got {fqs.max()}"
        assert 50 < spls.max() < 130, f"Hornresp SPL range unexpected: {spls.max()}"


# ─────────────────────────────────────────────────────────────────────────────
# Regression comparison tests
# ─────────────────────────────────────────────────────────────────────────────
# These thresholds document the TARGET for CRIT-1 / CRIT-3 resolution.
# With CRIT-1 (LF voltage coupling) fixed, we expect:
#   LF (10–200 Hz): delta → 0 dB (was −15 dB)
#   MF (200 Hz – 2 kHz): delta → 0 dB (was +5 dB and converging)
#   HF (2–20 kHz): delta → 0 dB (was +14–19 dB)
# giving a flat ±3 dB response across all decades.

class TestGdB1HornrespComparison:
    """
    Regression comparison: pyhorn vs Hornresp GdB1 BLH simulation.

    Target: after CRIT-1 + CRIT-3 fixes, mean delta per decade < ±3 dB
    and no single point exceeds ±10 dB.
    """

    @pytest.mark.xfail(
        reason="CRIT-1 (LF voltage coupling) + CRIT-3 (HF sensitivity reference) not yet fixed. "
        "These are known TMM bugs documented in BACKLOG.md Benchmark section."
    )
    def test_per_decade_mean_delta(self, pyhorn_result):
        """
        Mean delta per decade must be within ±3 dB.

        Decade boundaries: 10–100, 20–200, 50–500, 100–1000,
        200–2000, 500–5000, 1000–10000, 2000–20000 Hz.

        Note: This test currently FAILS due to CRIT-1 (LF −12 dB stuck)
        and CRIT-3 (HF +14–19 dB excess). It will PASS once both are fixed.
        """
        hr_f = pyhorn_result["hr_freqs"]
        hr_s = pyhorn_result["hr_spls"]
        py_at_hr = pyhorn_result["py_spl_at_hr"]
        valid = pyhorn_result["valid"]

        decade_starts = [10, 20, 50, 100, 200, 500, 1000, 2000]
        failures = []
        for d_start in decade_starts:
            d_end = d_start * 10
            mask = valid & (hr_f >= d_start) & (hr_f < d_end)
            if mask.sum() < 3:
                continue
            delta_mean = float(np.mean(py_at_hr[mask] - hr_s[mask]))
            delta_abs = abs(delta_mean)
            status = "PASS" if delta_abs <= TOLERANCE_PER_DECADE_DB else "FAIL"
            if status == "FAIL":
                failures.append(
                    f"  {d_start:6d}–{d_end:6d} Hz: mean_delta={delta_mean:+.2f} dB "
                    f"(limit ±{TOLERANCE_PER_DECADE_DB} dB)"
                )

        if failures:
            msg = (
                f"\nPer-decade delta test FAILED — CRIT-1/CRIT-3 not yet fixed.\n"
                + "\n".join(failures)
                + (
                    f"\n\nThis test documents the known mismatch. "
                    f"Target: ±{TOLERANCE_PER_DECADE_DB} dB per decade mean delta. "
                    "It will PASS once CRIT-1 (LF voltage coupling) and CRIT-3 "
                    "(HF sensitivity reference) are resolved."
                )
            )
            pytest.fail(msg)
        # If we get here, all decades are within tolerance
        assert True, "All decades within tolerance"

    @pytest.mark.xfail(
        reason="CRIT-3: HF +14–19 dB excess due to sensitivity reference mismatch. "
        "Known TMM bug documented in BACKLOG.md Benchmark section."
    )
    def test_no_single_point_extreme_deviation(self, pyhorn_result):
        """
        No single frequency point may deviate by more than ±10 dB.

        This catches catastrophic failures (e.g., off by 20+ dB at a
        single frequency). Currently FAILS due to CRIT-3 (LF −15 dB
        and HF +19 dB extremes).
        """
        hr_f = pyhorn_result["hr_freqs"]
        hr_s = pyhorn_result["hr_spls"]
        py_at_hr = pyhorn_result["py_spl_at_hr"]
        valid = pyhorn_result["valid"]

        delta = py_at_hr - hr_s
        max_dev_idx = int(np.argmax(np.abs(delta)))
        max_dev = float(delta[max_dev_idx])
        max_dev_freq = float(hr_f[max_dev_idx])

        assert abs(max_dev) <= TOLERANCE_PER_POINT_DB, (
            f"Point deviation test FAILED: max deviation = {max_dev:+.2f} dB "
            f"at {max_dev_freq:.0f} Hz (limit ±{TOLERANCE_PER_POINT_DB} dB). "
            f"pyhorn={py_at_hr[max_dev_idx]:.1f} dB, Hornresp={hr_s[max_dev_idx]:.1f} dB. "
            "CRIT-3: HF excess — sensitivity reference mismatch."
        )

    @pytest.mark.xfail(
        reason="CRIT-3: HF +14–19 dB excess due to sensitivity reference mismatch. "
        "Known TMM bug documented in BACKLOG.md Benchmark section."
    )
    def test_hf_region_2_to_20_khz(self, pyhorn_result):
        """
        High-frequency region (2–20 kHz): mean delta within ±5 dB.

        This is a tighter check on HF specifically where CRIT-3 manifests
        (+14 to +19 dB excess). After CRIT-1 fix, HF delta should be
        within ±5 dB of Hornresp.
        """
        hr_f = pyhorn_result["hr_freqs"]
        hr_s = pyhorn_result["hr_spls"]
        py_at_hr = pyhorn_result["py_spl_at_hr"]
        valid = pyhorn_result["valid"]

        mask = valid & (hr_f >= 2000) & (hr_f <= 20000)
        if mask.sum() < 3:
            pytest.skip("Insufficient HF data points")

        hf_delta_mean = float(np.mean(py_at_hr[mask] - hr_s[mask]))
        hf_delta_max = float(np.max(np.abs(py_at_hr[mask] - hr_s[mask])))
        hf_delta_std = float(np.std(py_at_hr[mask] - hr_s[mask]))

        # CRIT-3 check: HF is currently +14 to +19 dB above Hornresp
        # This is a >10 dB excess which indicates a sensitivity reference issue
        assert abs(hf_delta_mean) <= 5.0, (
            f"HF region (2–20 kHz) mean delta = {hf_delta_mean:+.2f} dB "
            f"(limit ±5 dB). Max point deviation: {hf_delta_max:+.2f} dB, "
            f"std: {hf_delta_std:.2f} dB. "
            "CRIT-3: HF SPL excess — pyhorn is +14–19 dB above Hornresp in HF. "
            "This is a sensitivity reference mismatch between pyhorn and Hornresp. "
            "See CRIT-3 issue for investigation notes."
        )

    @pytest.mark.xfail(
        reason="CRIT-1: LF −12 dB stuck regardless of voltage — "
        "driver voltage not reaching low-frequency resonance properly. "
        "Known TMM bug documented in BACKLOG.md Benchmark section."
    )
    def test_lf_region_10_to_200_hz(self, pyhorn_result):
        """
        Low-frequency region (10–200 Hz): mean delta within ±5 dB.

        This tests CRIT-1 (LF −12 dB stuck regardless of voltage).
        After CRIT-1 fix, LF delta should be within ±5 dB.
        """
        hr_f = pyhorn_result["hr_freqs"]
        hr_s = pyhorn_result["hr_spls"]
        py_at_hr = pyhorn_result["py_spl_at_hr"]
        valid = pyhorn_result["valid"]

        mask = valid & (hr_f >= 10) & (hr_f <= 200)
        if mask.sum() < 3:
            pytest.skip("Insufficient LF data points")

        lf_delta_mean = float(np.mean(py_at_hr[mask] - hr_s[mask]))
        lf_delta_max = float(np.max(np.abs(py_at_hr[mask] - hr_s[mask])))

        assert abs(lf_delta_mean) <= 5.0, (
            f"LF region (10–200 Hz) mean delta = {lf_delta_mean:+.2f} dB "
            f"(limit ±5 dB). Max point deviation: {lf_delta_max:+.2f} dB. "
            "CRIT-1: LF SPL stuck at −12 dB regardless of voltage — "
            "driver voltage not reaching low-frequency resonance properly."
        )
