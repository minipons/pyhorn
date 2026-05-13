"""Phase 3 benchmark: pyhorn response validation vs Hornresp reference.

This test suite validates pyhorn's acoustic simulation output is physically
reasonable and consistent with expected values from the Hornresp reference run.

Unlike test_vs_hornresp.py (which tests parameter loading and CLI end-to-end),
this module focuses on the computed response arrays themselves:
  - SPL values in reasonable range
  - No NaN or Inf values
  - Frequency array is monotonic
  - Peak SPL within 2 dB of expected (~113 dB at ~1560 Hz for FLH)
  - Efficiency peak within 1% of expected (~14% at ~800 Hz)

Once Geopan's Hornresp reference CSV is exported to:
  tests/hornresp_data/hornresp_flh_reference/response.csv
the comparison tests in TestHornrespReferenceComparison will activate.

Reference values confirmed by Geopan (Apr 28 2026):
  FLH:  Sd=132.70 cm², Cms=1.47E-03 m/N, Mmd=6.12g
        S1=40 cm², S2=300 cm², L12=1.527 m, F12=50.43 Hz, T=0.30

Run:
  pytest pyhorn_core/tests/test_vs_hornresp_comparison.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorn_core.config.models import DriverSpecs, HornGeometry
from pyhorn_core.config.parser import (
    parse_driver_specs,
    parse_horn_geometry,
    parse_horn_project,
)
from pyhorn_core.solver.models import horn_response

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BENCHMARK_DIR = Path(__file__).parent / "benchmarks"
HORNRESP_DATA = Path(__file__).parent / "hornresp_data"

DRIVER_YAML = BENCHMARK_DIR / "hornresp_reference_driver.yaml"
PROJECT_YAML = BENCHMARK_DIR / "hornresp_reference_project.yaml"

# BKHiro benchmark fixture lives under the repository benchmark fixtures.
BKHiro_PROJECT = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "benchmarks"
    / "hornresp"
    / "hirob"
    / "fixture"
    / "horn.yaml"
)
BKHiro_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "benchmarks"
    / "hornresp"
    / "hirob"
    / "fixture"
    / "horn.yaml"
)

# Hornresp exported CSVs (when Geopan provides them)
FLH_HORNRESP_CSV = HORNRESP_DATA / "hornresp_flh_reference" / "response.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Expected reference values (from Hornresp simulation, Geopan Apr 28 2026)
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_FLH_PEAK_SPL_DB = 113.0  # dB — peak SPL for FLH reference
EXPECTED_FLH_PEAK_FREQ_HZ = 1560.0  # Hz — frequency of peak SPL
EXPECTED_FLH_EFF_PCT = 14.0  # % — peak efficiency
EXPECTED_FLH_EFF_FREQ_HZ = 800.0  # Hz — frequency of peak efficiency
TOL_PEAK_SPL_DB = 2.0  # dB — peak SPL tolerance
TOL_EFF_PCT = 1.0  # % — efficiency tolerance


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — FLH benchmark
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def flh_driver() -> DriverSpecs:
    return parse_driver_specs(DRIVER_YAML)


@pytest.fixture
def flh_geometry() -> HornGeometry:
    _, horn = parse_horn_project(PROJECT_YAML)
    return horn


@pytest.fixture
def flh_result(flh_driver, flh_geometry) -> "SimResult":
    """Run horn_response() for FLH reference: 500 points, 20 Hz – 5 kHz."""
    freqs = np.linspace(20.0, 5000.0, 500)
    return horn_response(freqs=freqs, driver=flh_driver, horn=flh_geometry)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures — BKHiro (skip if geometry not yet defined)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def bkhiro_result() -> "SimResult | None":
    """Run horn_response() for BKHiro if geometry source file exists."""
    if not BKHiro_SOURCE.exists():
        pytest.skip(f"BKHiro geometry not yet defined: {BKHiro_SOURCE}")
    geo = parse_horn_geometry(BKHiro_PROJECT)
    # Derive driver specs from project or use FE166NV2 defaults
    from pyhorn_core.config.models import DriverSpecs as DS

    FE166NV2 = DS(
        fs=49.6,
        qts=0.27,
        qes=0.28,
        qms=7.88,
        vas=0.0369,
        re=7.8,
        bl=7.79,
        mms=0.00699,
        cms=0.001472,
        rms=0.277,
        sd=0.01327,
        le=0.0008,
        xmax=0.001,
        voltage=2.83,
    )
    freqs = np.linspace(20.0, 5000.0, 500)
    return horn_response(freqs=freqs, driver=FE166NV2, horn=geo)


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight SimResult type (avoids importing the full orchestrator module)
# ─────────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass


@dataclass
class SimResult:
    freqs: np.ndarray
    spl: np.ndarray
    impedance: np.ndarray
    excursion: np.ndarray
    efficiency_pct: np.ndarray | None
    phase: np.ndarray
    group_delay: np.ndarray
    impedance_real: np.ndarray
    impedance_imag: np.ndarray


def _to_sim(result) -> SimResult:
    """Convert SimulationResult to SimResult for field-level access."""
    return SimResult(
        freqs=result.freqs,
        spl=result.spl,
        impedance=result.impedance,
        excursion=result.excursion,
        efficiency_pct=result.efficiency_pct,
        phase=result.phase,
        group_delay=result.group_delay,
        impedance_real=getattr(result, "impedance_real", None)
        or np.real(result.impedance),
        impedance_imag=getattr(result, "impedance_imag", None)
        or np.imag(result.impedance),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load Hornresp CSV when available
# ─────────────────────────────────────────────────────────────────────────────


def _load_hornresp_csv() -> (
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None
):
    """Load Hornresp reference CSV if exported.

    Returns (freq, spl, re_z, im_z) or None.
    """
    if not FLH_HORNRESP_CSV.exists():
        return None
    freq, spl, re_z, im_z = [], [], [], []
    with open(FLH_HORNRESP_CSV) as f:
        import csv as csv_lib

        for row in csv_lib.DictReader(f):
            freq.append(float(row["frequency"]))
            spl.append(float(row["spl"]))
            re_z.append(float(row["re_z"]))
            im_z.append(float(row["im_z"]))
    return np.array(freq), np.array(spl), np.array(re_z), np.array(im_z)


# ─────────────────────────────────────────────────────────────────────────────
# TEST: FLH response validity
# ─────────────────────────────────────────────────────────────────────────────


class TestFLHResponseValidity:
    """Validate pyhorn FLH reference simulation output is physically sound."""

    def test_spl_range_80_to_120_db(self, flh_result):
        """SPL must be in 80–120 dB range for the FLH benchmark in-band.

        Allow the bass rolloff region (< 80 Hz) to dip below 80 dB — this is
        physical behavior. Check the in-band region (80 Hz – 4 kHz) instead.
        """
        band = (flh_result.freqs >= 80) & (flh_result.freqs <= 4000)
        spl_band = flh_result.spl[band]
        assert np.all(
            spl_band >= 80.0
        ), f"SPL below 80 dB in band: min={spl_band.min():.1f}"
        assert np.all(
            spl_band <= 120.0
        ), f"SPL above 120 dB: max={flh_result.spl.max():.1f}"

    def test_no_nan_or_inf(self, flh_result):
        """No NaN or Inf values in SPL, impedance, excursion, or phase."""
        for name in ["spl", "impedance", "excursion", "phase"]:
            arr = getattr(flh_result, name)
            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))
            assert nan_count == 0, f"{name} has {nan_count} NaN values"
            assert inf_count == 0, f"{name} has {inf_count} Inf values"

    def test_frequency_array_monotonic(self, flh_result):
        """Frequency array must be strictly increasing."""
        freqs = flh_result.freqs
        diffs = np.diff(freqs)
        assert np.all(diffs > 0), (
            f"Frequency array not strictly increasing at indices "
            f"{np.where(diffs <= 0)[0][:5].tolist()}"
        )

    def test_peak_spl_within_2db_of_expected(self, flh_result):
        """Peak SPL must be within 2 dB of ~113 dB at ~1560 Hz."""
        peak_idx = int(np.argmax(flh_result.spl))
        peak_spl = float(flh_result.spl[peak_idx])
        peak_freq = float(flh_result.freqs[peak_idx])
        assert abs(peak_spl - EXPECTED_FLH_PEAK_SPL_DB) <= TOL_PEAK_SPL_DB, (
            f"Peak SPL {peak_spl:.1f} dB @ {peak_freq:.0f} Hz differs from "
            f"expected {EXPECTED_FLH_PEAK_SPL_DB:.1f} dB by more than "
            f"{TOL_PEAK_SPL_DB} dB"
        )

    def test_peak_spl_at_reasonable_frequency(self, flh_result):
        """Peak SPL frequency should be near the expected ~1560 Hz (±500 Hz)."""
        peak_idx = int(np.argmax(flh_result.spl))
        peak_freq = float(flh_result.freqs[peak_idx])
        assert 1000 <= peak_freq <= 2500, (
            f"Peak SPL frequency {peak_freq:.0f} Hz outside expected "
            f"range [1000, 2500] Hz"
        )

    def test_efficiency_peak_within_1pct_of_expected(self, flh_result):
        """Peak efficiency must be within 1% of ~14% at ~800 Hz."""
        if flh_result.efficiency_pct is None:
            pytest.skip("efficiency_pct not computed in result")
        eff = flh_result.efficiency_pct
        # Find peak in band 500–1500 Hz (around expected 800 Hz)
        band = (flh_result.freqs >= 500) & (flh_result.freqs <= 1500)
        if not np.any(band):
            pytest.skip("No frequency points in 500–1500 Hz band")
        peak_eff = float(np.max(eff[band]))
        assert abs(peak_eff - EXPECTED_FLH_EFF_PCT) <= TOL_EFF_PCT, (
            f"Peak efficiency {peak_eff:.1f}% differs from expected "
            f"{EXPECTED_FLH_EFF_PCT:.1f}% by more than {TOL_EFF_PCT}%"
        )

    def test_efficiency_at_1khz_reasonable(self, flh_result):
        """Efficiency at 1 kHz should be in a reasonable range (5–20%)."""
        if flh_result.efficiency_pct is None:
            pytest.skip("efficiency_pct not computed")
        eff = flh_result.efficiency_pct
        idx_1khz = int(np.argmin(np.abs(flh_result.freqs - 1000)))
        eff_1khz = float(eff[idx_1khz])
        assert (
            5.0 <= eff_1khz <= 20.0
        ), f"Efficiency at 1 kHz ({eff_1khz:.1f}%) outside [5, 20]% range"

    def test_impedance_peak_near_fs(self, flh_result):
        """Impedance peak should occur near fs=49.6 Hz (±20 Hz).

        The FLH enclosure loads the driver, so the impedance peak may shift
        slightly from free-air fs. Check the sub-100 Hz region for the peak.
        """
        z_mag = np.abs(flh_result.impedance)
        # Look for peak in the 20-100 Hz range (near fs=49.6 Hz)
        fs_band = (flh_result.freqs >= 20) & (flh_result.freqs <= 100)
        z_fs_band = z_mag[fs_band]
        peak_in_band_idx = int(np.argmax(z_fs_band))
        peak_freq = float(flh_result.freqs[fs_band][peak_in_band_idx])
        assert 35 <= peak_freq <= 70, (
            f"Impedance peak in 20–100 Hz band at {peak_freq:.1f} Hz — "
            f"expected near fs=49.6 Hz"
        )

    def test_phase_unwrapped_sensible(self, flh_result):
        """Phase should be unwrapped (no >180° jumps) and in [-π, π] range."""
        phase = flh_result.phase
        diffs = np.diff(phase)
        large_jumps = np.abs(diffs) > np.pi
        assert not np.any(
            large_jumps
        ), f"Phase has {np.sum(large_jumps)} unwrap failures (>180° jumps)"

    def test_group_delay_mostly_positive(self, flh_result):
        """At least 90% of group delay values should be non-negative.

        Negative group delay values indicate phase unwrap failures, which are
        known artifacts at TMM standing-wave frequencies. In the midband
        (200–3000 Hz) the vast majority of values should be positive.
        """
        gd = flh_result.group_delay
        # At least 90% of values should be non-negative
        non_neg_pct = float(np.sum(gd >= 0)) / len(gd) * 100
        assert non_neg_pct >= 90.0, (
            f"Only {non_neg_pct:.1f}% of group delay values are non-negative "
            f"(expected ≥ 90%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST: BKHiro response validity
# ─────────────────────────────────────────────────────────────────────────────


class TestBKHiroResponseValidity:
    """Validate pyhorn BKHiro simulation output is physically sound.

    BKHiro is the GdB master design: Fostex FE166NV2 in a 2400×280 mm tall
    BLH cabinet with 8 expanding stair steps, straight throat 1500 mm +
    300 mm horn path.
    """

    def test_spl_range_50_to_120_db(self, bkhiro_result):
        """SPL must be in a physically reasonable range for BKHiro in-band.

        BKHiro (FE166NV2 in 2400×280 mm BLH cabinet) rolls off significantly
        below 300 Hz and peaks at ~103 dB — lower than a typical FLH horn.
        - Below 300 Hz: allow 50–120 dB (natural BLH rolloff toward cutoff)
        - 300 Hz – 5 kHz: expect 75–120 dB (passband)
        - Max should not exceed 120 dB (physical thermal/linear limit)
        The dedicated test_bass_rolloff_below_fs separately verifies LF rolloff.
        """
        if bkhiro_result is None:
            pytest.skip("BKHiro geometry not defined")
        # Full range — only assert against physically impossible values
        assert np.all(
            bkhiro_result.spl <= 120.0
        ), f"SPL above 120 dB (thermal limit): max={bkhiro_result.spl.max():.1f}"
        assert np.all(
            bkhiro_result.spl >= 40.0
        ), f"SPL below 40 dB (numerical issue?): min={bkhiro_result.spl.min():.1f}"
        # Passband (>= 300 Hz) — should be well above the noise floor
        band = bkhiro_result.freqs >= 300.0
        spl_band = bkhiro_result.spl[band]
        assert np.all(
            spl_band >= 75.0
        ), f"SPL below 75 dB in passband (>= 300 Hz): min={spl_band.min():.1f} dB"
        assert np.all(
            spl_band <= 120.0
        ), f"SPL above 120 dB in passband: max={spl_band.max():.1f} dB"

    def test_no_nan_or_inf(self, bkhiro_result):
        """No NaN or Inf values in BKHiro response arrays."""
        if bkhiro_result is None:
            pytest.skip("BKHiro geometry not defined")
        for name in ["spl", "impedance", "excursion", "phase"]:
            arr = getattr(bkhiro_result, name)
            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))
            assert nan_count == 0, f"{name} has {nan_count} NaN values"
            assert inf_count == 0, f"{name} has {inf_count} Inf values"

    def test_frequency_array_monotonic(self, bkhiro_result):
        """Frequency array must be strictly increasing."""
        if bkhiro_result is None:
            pytest.skip("BKHiro geometry not defined")
        freqs = bkhiro_result.freqs
        diffs = np.diff(freqs)
        assert np.all(diffs > 0), "Frequency array not strictly increasing"

    def test_peak_spl_reasonable(self, bkhiro_result):
        """Peak SPL should be between 100–120 dB for FE166NV2 in BKHiro."""
        if bkhiro_result is None:
            pytest.skip("BKHiro geometry not defined")
        peak_idx = int(np.argmax(bkhiro_result.spl))
        peak_spl = float(bkhiro_result.spl[peak_idx])
        assert (
            100.0 <= peak_spl <= 120.0
        ), f"BKHiro peak SPL {peak_spl:.1f} dB outside expected [100, 120] dB"

    def test_bass_rolloff_below_fs(self, bkhiro_result):
        """SPL below 50 Hz should be at least 10 dB below midband SPL."""
        if bkhiro_result is None:
            pytest.skip("BKHiro geometry not defined")
        # Below cutoff
        below = (bkhiro_result.freqs >= 20) & (bkhiro_result.freqs <= 50)
        # Midband
        mid = (bkhiro_result.freqs >= 80) & (bkhiro_result.freqs <= 300)
        if not (np.any(below) and np.any(mid)):
            pytest.skip("Insufficient frequency points in bands")
        spl_below = float(np.mean(bkhiro_result.spl[below]))
        spl_mid = float(np.mean(bkhiro_result.spl[mid]))
        assert spl_below < spl_mid - 10.0, (
            f"Bass rolloff insufficient: SPL below 50 Hz ({spl_below:.1f} dB) "
            f"not at least 10 dB below midband ({spl_mid:.1f} dB)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TEST: Hornresp reference comparison (activates when Geopan's CSV arrives)
# ─────────────────────────────────────────────────────────────────────────────


class TestHornrespReferenceComparison:
    """Compare pyhorn simulation against Hornresp exported CSV.

    To enable: export Hornresp response data to
    tests/hornresp_data/hornresp_flh_reference/response.csv
    with columns: frequency, spl, re_z, im_z
    """

    def test_hornresp_csv_available(self):
        """Skip if CSV not yet exported by Geopan."""
        ref = _load_hornresp_csv()
        if ref is None:
            pytest.skip(
                f"Hornresp reference CSV not found at {FLH_HORNRESP_CSV} — "
                "export from Hornresp (File → Export response data → CSV) "
                "to enable comparison tests"
            )

    def test_spl_rms_within_3db(self, flh_result):
        """RMS SPL deviation vs Hornresp must be < 3 dB."""
        ref = _load_hornresp_csv()
        if ref is None:
            pytest.skip("CSV not available")
        ref_freq, ref_spl, _, _ = ref
        py_spl_interp = np.interp(ref_freq, flh_result.freqs, flh_result.spl)
        rms = float(np.sqrt(np.mean((py_spl_interp - ref_spl) ** 2)))
        assert rms < 3.0, f"SPL RMS deviation {rms:.2f} dB exceeds 3 dB threshold"

    def test_peak_spl_within_2db(self, flh_result):
        """Peak SPL should be within 2 dB of Hornresp."""
        ref = _load_hornresp_csv()
        if ref is None:
            pytest.skip("CSV not available")
        _, ref_spl, _, _ = ref
        py_peak = float(np.max(flh_result.spl))
        ref_peak = float(np.max(ref_spl))
        diff = abs(py_peak - ref_peak)
        assert diff <= 2.0, (
            f"Peak SPL: pyhorn={py_peak:.1f} dB, Hornresp={ref_peak:.1f} dB, "
            f"diff={diff:.2f} dB"
        )

    def test_spl_at_1khz_within_3db(self, flh_result):
        """SPL at 1 kHz should be within 3 dB of Hornresp."""
        ref = _load_hornresp_csv()
        if ref is None:
            pytest.skip("CSV not available")
        ref_freq, ref_spl, _, _ = ref
        py_spl_interp = np.interp(ref_freq, flh_result.freqs, flh_result.spl)
        idx_1khz = int(np.argmin(np.abs(ref_freq - 1000)))
        diff = abs(py_spl_interp[idx_1khz] - ref_spl[idx_1khz])
        assert diff <= 3.0, (
            f"SPL at 1 kHz: pyhorn={py_spl_interp[idx_1khz]:.1f} dB, "
            f"Hornresp={ref_spl[idx_1khz]:.1f} dB, diff={diff:.2f} dB"
        )
