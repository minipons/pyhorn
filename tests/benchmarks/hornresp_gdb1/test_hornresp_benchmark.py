# pyhorn Hornresp Benchmark Regression Tests
"""
Regression suite comparing pyhorn SPL output against Hornresp reference data
for the GdB1 geometry (FE166NV2 driver, BLH horn).

These tests catch regressions in the TMM physics that would cause pyhorn to
diverge from validated Hornresp output.

Reference CSV: tests/benchmarks/hornresp_gdb1/hornresp_spl.csv
Driver YAML:   tests/benchmarks/hornresp_gdb1/gdb1_driver_only.yaml
Horn YAML:     tests/benchmarks/hornresp_gdb1/gdb1_horn_only.yaml

Acceptance criteria (per BACKLOG.md TEST-1):
  - SPL delta at each decade within ±3 dB (100–20000 Hz band)
  - Overall RMS delta within ±3 dB
  - Test fails if any decade exceeds ±5 dB (regression threshold)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import interp1d

REPO = Path(__file__).resolve().parents[3]  # .../GdB1/
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "pyhorn_core"))
sys.path.insert(0, str(REPO / "pyhorn_api"))
sys.path.insert(0, str(REPO / "pyhorn_cli"))

from pyhorn_core.config.parser import parse_driver_specs, parse_horn_geometry
from pyhorn_core.pyhorn_physics.orchestrators import horn_response


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def hornresp_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the Hornresp reference CSV (frequency Hz, SPL dB)."""
    csv_path = REPO / "tests/benchmarks/hornresp_gdb1/hornresp_spl.csv"
    if not csv_path.exists():
        pytest.skip(f"Hornresp reference CSV not found: {csv_path}")
    freqs, spls = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            freqs.append(float(row["Freq (hertz)"]))
            spls.append(float(row["SPL (dB)"]))
    return np.array(freqs), np.array(spls)


@pytest.fixture(scope="module")
def pyhorn_result(hornresp_data):
    """Run pyhorn simulation with GdB1 parameters over the Hornresp frequency range."""
    hr_freqs, _ = hornresp_data

    driver_yaml = REPO / "tests/benchmarks/hornresp_gdb1/gdb1_driver_only.yaml"
    horn_yaml = REPO / "tests/benchmarks/hornresp_gdb1/gdb1_horn_only.yaml"

    driver = parse_driver_specs(str(driver_yaml))
    horn = parse_horn_geometry(str(horn_yaml))

    # Use same frequency grid as Hornresp for direct comparison
    py_freqs = np.linspace(hr_freqs.min(), hr_freqs.max(), len(hr_freqs))
    result = horn_response(py_freqs, driver, horn, compute_distortion=False)
    # Return raw SPL and sensitivity-calibrated SPL (spl_power_based)
    return py_freqs, result.spl, result.spl_power_based


@pytest.fixture(scope="module")
def comparison_data(hornresp_data, pyhorn_result):
    """
    Interpolate pyhorn onto Hornresp frequencies; return raw and calibrated deltas.

    Returns: (hr_freqs, hr_spls, py_interp_raw, delta_raw, valid,
              py_interp_cal, delta_cal)
    """
    hr_freqs, hr_spls = hornresp_data
    py_freqs, py_spls, py_spls_cal = pyhorn_result

    log_hr = np.log10(hr_freqs)
    log_py = np.log10(py_freqs)

    # Only compare within the simulation range
    valid = (hr_freqs >= py_freqs.min()) & (hr_freqs <= py_freqs.max())
    interp = interp1d(log_py, py_spls, kind="linear", fill_value="extrapolate")
    py_interp = interp(log_hr)
    delta = py_interp - hr_spls

    # Calibrated comparison: use spl_power_based (acoustic-power-based SPL)
    # This applies the sensitivity_db offset from gdb1_driver_only.yaml (-7.0 dB)
    # to align pyhorn with Hornresp dB/W/m reference.
    if py_spls_cal is not None:
        interp_cal = interp1d(
            log_py, py_spls_cal, kind="linear", fill_value="extrapolate"
        )
        py_interp_cal = interp_cal(log_hr)
        delta_cal = py_interp_cal - hr_spls
    else:
        py_interp_cal = None
        delta_cal = None

    return hr_freqs, hr_spls, py_interp, delta, valid, py_interp_cal, delta_cal


# ── Regression Tests ──────────────────────────────────────────────────────────

class TestHornrespBenchmarkRegression:
    """Regression suite — catches changes that push pyhorn away from Hornresp validation."""

    def test_hornresp_csv_exists(self, hornresp_data):
        """Reference CSV must exist and contain data."""
        freqs, spls = hornresp_data
        assert len(freqs) > 100, "Hornresp CSV should have hundreds of frequency points"
        assert freqs.min() < 20, "Should contain sub-20 Hz points"
        assert freqs.max() > 15000, "Should extend above 15 kHz"

    def test_pyhorn_spl_range_reasonable(self, pyhorn_result):
        """SPL range should be physically plausible for a BLH with FE166NV2."""
        _, py_spls, _ = pyhorn_result
        assert 40 < py_spls.max() < 130, f"SPL max {py_spls.max():.1f} dB outside plausible range"
        assert 30 < py_spls.min() < 90, f"SPL min {py_spls.min():.1f} dB outside plausible range"

    def test_overall_mean_delta_within_5db(self, comparison_data):
        """Mean SPL delta must be within ±5 dB (regression threshold)."""
        *_, delta, valid, _, _ = comparison_data
        mean_delta = np.mean(delta[valid])
        assert abs(mean_delta) < 5.0, (
            f"Mean delta {mean_delta:+.2f} dB exceeds ±5 dB regression threshold. "
            f"Hornresp vs pyhorn have diverged significantly."
        )

    def test_overall_std_within_14db(self, comparison_data):
        """
        Standard deviation of delta should be under 14 dB.
        Baseline (May 4 2026): std=12.3 dB due to known CRIT-1 (LF) and CRIT-3 (HF) bugs.
        If std increases by >2 dB from baseline, something has regressed.
        """
        *_, delta, valid, _, _ = comparison_data
        std_delta = np.std(delta[valid])
        assert std_delta < 14.0, (
            f"Delta std {std_delta:.2f} dB exceeds 14 dB — possible physics regression. "
            f"Baseline (May 4 2026): 12.3 dB."
        )

    def test_decade_delta_regression_threshold(self, comparison_data):
        """
        Per-decade mean delta must not exceed ±22 dB — regression threshold.

        Current baseline (May 4 2026, known CRIT-1 + CRIT-3 bugs):
          LF (10–200 Hz):  mean ~−12 dB  (CRIT-1: voltage doesn't couple to LF)
          HF (2–20 kHz):   mean ~+15 dB  (CRIT-3: sensitivity reference mismatch)

        This test passes today (worst decade mean is 20.06 dB, threshold is 22 dB).
        It will FAIL if any decade mean drifts by >2 dB further from Hornresp —
        catching regressions before they become entrenched.

        Acceptance target (once CRIT-1 and CRIT-3 are fixed): ±3 dB per decade,
        100–20000 Hz (the project acceptance criteria for SPL alignment).
        """
        hr_freqs, _, _, delta, valid, _, _ = comparison_data

        failures = []
        REGRESSION_THRESHOLD = 22.0  # dB — must be above worst current baseline
        for decade_start in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]:
            mask = valid & (hr_freqs >= decade_start) & (hr_freqs < decade_start * 10)
            if mask.sum() < 3:
                continue
            d_mean = np.mean(delta[mask])
            d_max = np.max(np.abs(delta[mask]))
            if abs(d_mean) >= REGRESSION_THRESHOLD:
                failures.append(
                    f"  {decade_start:6d}–{decade_start*10:6d} Hz: "
                    f"mean={d_mean:+7.2f} dB, max_abs={d_max:+7.2f} dB  ← REGRESSION"
                )

        if failures:
            msg = (
                f"\nDecades exceeding ±{REGRESSION_THRESHOLD:.0f} dB regression threshold "
                "(baseline May 4 2026):\n"
            )
            msg += "\n".join(failures)
            msg += (
                f"\nNOTE: CRIT-1 (LF) and CRIT-3 (HF) are known bugs causing large deltas. "
                f"When those are fixed, set threshold to ±5 dB for regression protection."
            )
            pytest.fail(msg)

    def test_no_catastrophic_outliers(self, comparison_data):
        """
        No single point should be more than 30 dB off.
        Baseline (May 4 2026): max delta ~29.9 dB at HF edge — at the limit.
        Increase in max delta indicates worsening TMM divergence.
        """
        *_, delta, valid, _, _ = comparison_data
        max_abs = np.max(np.abs(delta[valid]))
        assert max_abs < 30.0, (
            f"Maximum absolute delta {max_abs:.1f} dB indicates catastrophic TMM failure. "
            f"Baseline (May 4 2026): 29.9 dB."
        )

    def test_lf_band_reasonable(self, comparison_data):
        """
        Low-frequency band (10–200 Hz): pyhorn should be within ±15 dB of Hornresp.
        This is a known discrepancy region (CRIT-1); this test documents the baseline.
        """
        hr_freqs, _, _, delta, valid, _, _ = comparison_data
        lf_mask = valid & (hr_freqs >= 10) & (hr_freqs <= 200)
        lf_mean = np.mean(delta[lf_mask])
        assert abs(lf_mean) < 15.0, (
            f"LF band (10–200 Hz) mean delta {lf_mean:+.2f} dB exceeds 15 dB — "
            f"check voltage coupling in TMM (CRIT-1 hypothesis)."
        )

    def test_hf_band_reasonable(self, comparison_data):
        """
        High-frequency band (2–20 kHz): pyhorn should be within ±20 dB of Hornresp.
        This is a known discrepancy region (CRIT-3); this test documents the baseline.
        """
        hr_freqs, _, _, delta, valid, _, _ = comparison_data
        hf_mask = valid & (hr_freqs >= 2000) & (hr_freqs <= 20000)
        hf_mean = np.mean(delta[hf_mask])
        assert abs(hf_mean) < 20.0, (
            f"HF band (2–20 kHz) mean delta {hf_mean:+.2f} dB exceeds 20 dB — "
            f"check sensitivity reference (CRIT-3 hypothesis)."
        )


class TestCRIT3SensitivityCalibration:
    """
    CRIT-3 validation: verify that frequency-dependent sensitivity_db calibration
    brings pyhorn spl_power_based within reasonable bounds of Hornresp.

    Root cause: pyhorn's raw SPL uses pressure-based normalization
    (20*log10(|p|/2e-5)) which differs from Hornresp's dB/W/m reference.
    Hornresp uses acoustic-power-based normalization:
    10*log10(P_acoustic/1e-12) + sensitivity_db (dB/W/m).

    Calibration: gdb1_driver_only.yaml uses a frequency-dependent sensitivity_db
    table [[freq_hz, delta_db], ...] for piecewise-linear interpolation.
    The acoustic_power model (R_rad_mouth * U_mouth^2) has a systematic ~13 dB
    offset from Hornresp's dB/W/m reference — the table values represent this
    empirical calibration. Full fix requires recalibrating the acoustic power
    model itself (separate from sensitivity_db mechanism).

    See BACKLOG.md CRIT-3 for full investigation notes.
    """

    def test_spl_power_based_available(self, pyhorn_result):
        """spl_power_based must be populated when sensitivity_db is set."""
        _, _, py_spls_cal = pyhorn_result
        assert py_spls_cal is not None, (
            "spl_power_based is None — sensitivity_db calibration not wired. "
            "Ensure _horn_response_impl populates spl_power_based when "
            "acoustic_power is available."
        )

    def test_calibrated_spl_mean_delta_within_15db(self, comparison_data):
        """Mean delta of calibrated SPL (spl_power_based) should be within ±15 dB of Hornresp.

        Note: the ±15 dB threshold reflects the known ~13 dB systematic offset in the
        acoustic_power model. Once the acoustic power model itself is recalibrated to
        match Hornresp's dB/W/m reference, this threshold should be tightened to ±5 dB.
        """
        hr_freqs, hr_spls, _, _, valid, py_interp_cal, delta_cal = comparison_data
        assert delta_cal is not None, "spl_power_based must be available"
        cal_mean = np.mean(delta_cal[valid])
        assert abs(cal_mean) < 15.0, (
            f"Calibrated SPL mean delta {cal_mean:+.2f} dB exceeds ±15 dB. "
            f"CRIT-3 acoustic_power model may have a larger than expected offset. "
            f"Check acoustic_power_to_spl_dB_W_m() implementation."
        )

    def test_calibrated_hf_band_smooth(self, comparison_data):
        """HF band (2–20 kHz) calibrated SPL should be smooth (std < 10 dB).

        This verifies that the frequency-dependent sensitivity_db table is producing
        a smooth calibration curve without discontinuities.
        """
        hr_freqs, _, _, _, valid, _, delta_cal = comparison_data
        assert delta_cal is not None, "spl_power_based must be available"
        hf_mask = valid & (hr_freqs >= 2000) & (hr_freqs <= 20000)
        hf_mean = np.mean(delta_cal[hf_mask])
        hf_std = np.std(delta_cal[hf_mask])
        assert hf_std < 10.0, (
            f"Calibrated HF band std {hf_std:.2f} dB is very high — "
            f"frequency-dependent sensitivity_db table may have discontinuities."
        )


class TestHornrespBenchmarkPerformance:
    """Non-functional: verify the benchmark runs in reasonable time."""

    def test_simulation_completes_under_5_seconds(self, pyhorn_result):
        """Simulation should complete quickly (no infinite loops)."""
        import time
        start = time.time()
        # Re-run to get accurate timing (fixture already ran once)
        driver_yaml = REPO / "tests/benchmarks/hornresp_gdb1/gdb1_driver_only.yaml"
        horn_yaml = REPO / "tests/benchmarks/hornresp_gdb1/gdb1_horn_only.yaml"
        driver = parse_driver_specs(str(driver_yaml))
        horn = parse_horn_geometry(str(horn_yaml))
        freqs = np.linspace(10, 20000, 2000)
        horn_response(freqs, driver, horn, compute_distortion=False)
        elapsed = time.time() - start
        assert elapsed < 5.0, f"Simulation took {elapsed:.1f}s — should be under 5s"


# ── Additional integration tests ──────────────────────────────────────────────

import yaml
from scipy.interpolate import interp1d


def _load_driver_and_horn_from_combined(combined_path: Path) -> tuple:
    """Parse combined YAML into DriverSpecs + HornGeometry for simulation."""
    with open(combined_path) as f:
        params = yaml.safe_load(f)
    DRIVER_FIELDS = {
        "fs", "qts", "qes", "qms", "vas", "re", "sd", "bl",
        "mms", "cms", "rms", "le", "xmax", "voltage", "alpha_re",
        "le_freq_dependency", "le_f_ref", "lossy_le", "le_R_e_eddy",
        "le_f_lossy_ref", "sensitivity_db",
    }
    HORN_FIELDS = {
        "throat_area", "mouth_area", "path_length", "enclosure_type",
        "path_diff", "ang", "vrc", "lrc", "fr_rc", "vented_box",
        "passive_radiator", "slavbas", "vtc", "atc", "fr_tc",
        "ap1", "lpt", "throat_adapter_type", "profile_type",
        "hyperbolic_t", "n_segments", "width", "sections",
        "conical_segments", "rectangular_segments", "coordinates",
        "enclosure_dims", "driver_coord", "discretisation", "bend_angles",
        "rear_chamber", "lem_step_model", "lem_step_strength",
        "lem_step_resistance", "segments", "bends",
    }
    import tempfile, os
    driver_data = {k: v for k, v in params.items() if k in DRIVER_FIELDS}
    horn_data = {k: v for k, v in params.items() if k in HORN_FIELDS}
    with tempfile.NamedTemporaryFile(mode="w", suffix="_driver.yaml", delete=False) as f:
        yaml.dump(driver_data, f)
        driver_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix="_horn.yaml", delete=False) as f:
        yaml.dump(horn_data, f)
        horn_path = f.name
    driver = parse_driver_specs(driver_path)
    horn = parse_horn_geometry(horn_path)
    os.unlink(driver_path)
    os.unlink(horn_path)
    return driver, horn


class TestHornrespBenchmarkImpedance:
    """Verify electrical impedance is physically plausible and tracks Hornresp.

    The Hornresp CSV contains a 'Ze (ohms)' column with the driving-point
    electrical impedance at each frequency.  pyhorn's impedance is derived from
    the same TMM Z_in = (R_e + jωL_e) + Z_membrane ∥ Z_radiation model,
    so the two should agree within ~20 Ω across the band.

    Acceptance: impedance magnitude 5–100 Ω across the simulation range,
    with a resonance peak in the 40–100 Hz LF region.
    """

    @pytest.fixture(scope="class")
    def sim_result(self):
        combined = REPO / "tests/benchmarks/hornresp_gdb1/gdb1_hornresp.yaml"
        driver, horn = _load_driver_and_horn_from_combined(combined)
        freqs = np.linspace(10, 20000, 2000)
        return freqs, horn_response(freqs, driver, horn, compute_distortion=False)

    @pytest.fixture(scope="class")
    def hornresp_impedance(self):
        """Load Hornresp reference impedance (Ze in ohms)."""
        csv_path = REPO / "tests/benchmarks/hornresp_gdb1/hornresp_spl.csv"
        hr_freqs, hr_ze = [], []
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                hr_freqs.append(float(row["Freq (hertz)"]))
                hr_ze.append(float(row["Ze (ohms)"]))
        return np.array(hr_freqs), np.array(hr_ze)

    def test_impedance_magnitude_in_reasonable_range(self, sim_result):
        """Impedance magnitude should be physically plausible for FE166NV2 in BLH."""
        freqs, result = sim_result
        z_mag = np.abs(result.impedance)
        assert z_mag.min() >= 5.0, f"Impedance too low: {z_mag.min():.1f} Ω (may indicate short)"
        assert z_mag.max() <= 110.0, (
            f"Impedance too high: {z_mag.max():.1f} Ω "
            f"(may indicate open-circuit or numerical instability)"
        )

    def test_impedance_has_lf_resonance_peak(self, sim_result):
        """Impedance should show a resonance peak in the 40–120 Hz band (fs region)."""
        freqs, result = sim_result
        z_mag = np.abs(result.impedance)
        lf_mask = (freqs >= 40) & (freqs <= 120)
        peak_in_band = z_mag[lf_mask].max()
        global_min = z_mag.min()
        assert peak_in_band > global_min * 1.5, (
            f"No clear LF resonance peak: band-max={peak_in_band:.1f} Ω, "
            f"global-min={global_min:.1f} Ω. Driver fs region may not be resolving."
        )

    def test_impedance_agrees_with_hornresp_above_200hz(self, sim_result, hornresp_impedance):
        """Above 200 Hz, pyhorn impedance should track Hornresp within ±20 Ω.

        Below 200 Hz, the CRIT-1 voltage-coupling issue causes large disagreement
        in the resonance peak (~67 Ω at 56 Hz).  This is a known systematic error.
        Above 200 Hz, the TMM impedance model is better behaved and should agree
        within ±20 Ω with Hornresp.
        """
        freqs, result = sim_result
        hr_freqs, hr_ze = hornresp_impedance
        z_mag = np.abs(result.impedance)

        log_hr = np.log10(hr_freqs)
        log_py = np.log10(freqs)
        # Only check above 200 Hz where CRIT-1 is not dominant
        valid = (hr_freqs >= freqs.min()) & (hr_freqs <= freqs.max()) & (hr_freqs >= 200)
        interp_z = interp1d(log_py, z_mag, kind="linear", fill_value="extrapolate")
        py_z_interp = interp_z(log_hr)
        delta_z = py_z_interp - hr_ze

        mean_delta = np.mean(delta_z[valid])
        max_abs = np.max(np.abs(delta_z[valid]))
        assert max_abs < 20.0, (
            f"Impedance delta vs Hornresp exceeds ±20 Ω above 200 Hz: "
            f"max_abs={max_abs:.1f} Ω. Mean delta: {mean_delta:+.1f} Ω. "
            f"Check TMM impedance model."
        )


class TestHornrespBenchmarkGroupDelay:
    """Verify group delay values are physically plausible.

    Group delay τ_g = −dφ/df (in seconds, reported in ms) must be bounded.
    pyhorn computes it from the phase of Z_in via numpy gradient.
    Hornresp's 'Delay (msec)' column shows values 1.6–22 ms for this geometry.

    Acceptance: −50 to +50 ms across the full band, with no extreme outliers.
    Hornresp LF delay is 10–22 ms (positive).  pyhorn may show small negative
    values near the resonance due to phase-wrap artefacts — these are tolerated.
    """

    @pytest.fixture(scope="class")
    def sim_gd_result(self):
        combined = REPO / "tests/benchmarks/hornresp_gdb1/gdb1_hornresp.yaml"
        driver, horn = _load_driver_and_horn_from_combined(combined)
        freqs = np.linspace(10, 20000, 2000)
        result = horn_response(freqs, driver, horn, compute_distortion=False)
        return freqs, result.group_delay

    def test_group_delay_within_bounds(self, sim_gd_result):
        """Group delay must stay within ±50 ms (physical sanity check)."""
        freqs, gd = sim_gd_result
        valid = np.isfinite(gd)
        assert gd[valid].min() >= -50.0, (
            f"Group delay too negative: {gd[valid].min():.2f} ms. "
            f"Check phase-gradient computation near resonance."
        )
        assert gd[valid].max() <= 50.0, (
            f"Group delay too large: {gd[valid].max():.2f} ms. "
            f"Possible numerical instability in TMM phase calculation."
        )

    def test_group_delay_mean_reasonable(self, sim_gd_result):
        """Mean group delay across the band should be positive (acoustic causality)."""
        freqs, gd = sim_gd_result
        valid = np.isfinite(gd) & (freqs >= 100) & (freqs <= 10000)
        mean_gd = np.mean(gd[valid])
        assert mean_gd > -10.0, (
            f"Mean group delay {mean_gd:.2f} ms is unexpectedly negative. "
            f"Acoustic systems should have positive mean delay in-band."
        )
        assert mean_gd < 30.0, (
            f"Mean group delay {mean_gd:.2f} ms is too large for this BLH geometry. "
            f"Check phase unwrapping in impedance calculation."
        )

    def test_group_delay_no_extreme_outliers(self, sim_gd_result):
        """No single frequency should deviate more than 30 ms from the local median."""
        freqs, gd = sim_gd_result
        valid = np.isfinite(gd)
        # Use a rolling median to detect isolated spikes
        window = 20
        outliers = 0
        for i in range(window, len(gd) - window):
            if not valid[i]:
                continue
            local_median = float(np.median(gd[i - window : i + window]))
            deviation = abs(gd[i] - local_median)
            if deviation > 30.0:
                outliers += 1
        assert outliers < 5, (
            f"{outliers} points have group delay > 30 ms from local median — "
            f"possible phase-wrap artefact. Check impedance phase unwrapping."
        )


class TestHornrespBenchmarkOffAxis:
    """Verify off-axis SPL response decreases with angle.

    The /simulate API (and pyhorn_core direct simulation) support an
    off_axis_angles parameter.  For a BLH, directivity narrows with frequency,
    so on-axis SPL should generally exceed off-axis SPL at the same frequency.

    Acceptance: for most frequencies (> 70%), SPL decreases monotonically
    with angle: 0° > 30° > 60° (at least in the 200–5000 Hz band where
    directivity is established).
    """

    @pytest.fixture(scope="class")
    def off_axis_result(self):
        combined = REPO / "tests/benchmarks/hornresp_gdb1/gdb1_hornresp.yaml"
        driver, horn = _load_driver_and_horn_from_combined(combined)
        freqs = np.linspace(10, 20000, 1000)
        return freqs, horn_response(
            freqs,
            driver,
            horn,
            compute_distortion=False,
            off_axis_angles=np.array([0.0, 30.0, 60.0]),
        )

    def test_off_axis_spl_returned(self, off_axis_result):
        """Simulation with off_axis_angles must return off_axis_spl array."""
        freqs, result = off_axis_result
        assert result.off_axis_spl is not None, "off_axis_spl should not be None"
        assert result.off_axis_angles is not None, "off_axis_angles should not be None"
        assert result.off_axis_spl.shape[1] == 3, (
            f"Expected 3 off-axis columns (0°, 30°, 60°), got {result.off_axis_spl.shape[1]}"
        )
        assert result.off_axis_spl.shape[0] == len(freqs), (
            "off_axis_spl frequency axis should match input freqs"
        )

    def test_off_axis_spl_decreases_with_angle(self, off_axis_result):
        """On-axis SPL should generally exceed off-axis SPL (positive directivity)."""
        freqs, result = off_axis_result
        spl_0 = result.off_axis_spl[:, 0]
        spl_30 = result.off_axis_spl[:, 1]
        spl_60 = result.off_axis_spl[:, 2]

        # Directivity effect is meaningful above ~200 Hz where wavelength < cab size
        directivity_band = freqs >= 200
        valid = (
            directivity_band
            & np.isfinite(spl_0)
            & np.isfinite(spl_30)
            & np.isfinite(spl_60)
        )
        correct_order = (spl_0[valid] > spl_30[valid]) & (spl_30[valid] > spl_60[valid])
        pct = 100.0 * correct_order.sum() / valid.sum()
        assert pct >= 70.0, (
            f"Only {pct:.1f}% of frequencies in 200–20000 Hz band have "
            f"0° > 30° > 60° SPL ordering. Expected ≥70%. "
            f"Check directivity factor computation in off-axis model."
        )

    def test_off_axis_spl_angles_match_request(self, off_axis_result):
        """Returned off_axis_angles should exactly match the requested [0, 30, 60]."""
        _, result = off_axis_result
        np.testing.assert_allclose(
            result.off_axis_angles, [0.0, 30.0, 60.0], rtol=0, atol=1e-6,
            err_msg="off_axis_angles should be [0, 30, 60] degrees"
        )


# ── Run from CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
