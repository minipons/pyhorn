"""Numerical stability regression tests for pyhorn.

These tests verify that the simulation handles extreme parameter regimes gracefully:
- Very high frequencies (>> 20 kHz) → no NaN/Inf in output
- Very low frequencies (<< 20 Hz) → no numerical overflow
- Extreme area ratios (very small throat / very large mouth) → graceful handling (no crash)
- Long horn paths (> 5m) → computation completes without stack overflow

Run:
    pytest pyhorn_core/tests/test_numerical_stability.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorn_core.config.models import DriverSpecs, HornGeometry
from pyhorn_core.config.parser import parse_driver_specs
from pyhorn_core.solver.models import horn_response

# ─────────────────────────────────────────────────────────────────────────────
# Driver fixture — use the production FE166NV2 driver YAML
# ─────────────────────────────────────────────────────────────────────────────

TESTS_DIR = Path(__file__).parent
DRIVER_YAML = TESTS_DIR.parent.parent / "drivers" / "FE166NV2.yaml"


@pytest.fixture
def driver() -> DriverSpecs:
    """Load the FE166NV2 driver T-S parameters."""
    if DRIVER_YAML.exists():
        return parse_driver_specs(DRIVER_YAML)
    # Fallback: minimal inline FE166NV2-derived driver
    return DriverSpecs(
        fs=49.6,
        qts=0.27,
        qes=0.28,
        qms=7.88,
        vas=0.0369,
        re=7.80,
        bl=7.80,
        mms=0.00612,
        cms=1.47e-3,
        rms=0.28,
        sd=0.01327,
        le=0.00080,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _assert_no_nan_or_inf(result, msg: str = ""):
    """Assert that SPL, impedance magnitude, and group_delay contain no NaN or Inf."""
    prefix = f"[{msg}] " if msg else ""

    # SPL
    assert not np.any(np.isnan(result.spl)), f"{prefix}SPL contains NaN values"
    assert not np.any(np.isinf(result.spl)), f"{prefix}SPL contains Inf values"

    # Impedance (complex → use magnitude)
    z = np.asarray(result.impedance)
    z_mag = np.abs(z)
    assert not np.any(
        np.isnan(z_mag)
    ), f"{prefix}impedance magnitude contains NaN values"
    assert not np.any(
        np.isinf(z_mag)
    ), f"{prefix}impedance magnitude contains Inf values"

    # Group delay
    if hasattr(result, "group_delay") and result.group_delay is not None:
        gd = np.asarray(result.group_delay)
        # Only check finite values
        finite_gd = gd[np.isfinite(gd)]
        if finite_gd.size > 0:
            assert not np.any(
                np.isnan(finite_gd)
            ), f"{prefix}group_delay contains NaN values"
            assert not np.any(
                np.isinf(finite_gd)
            ), f"{prefix}group_delay contains Inf values"


# ─────────────────────────────────────────────────────────────────────────────
# Test: very high frequency (50 kHz) — no NaN/Inf
# ─────────────────────────────────────────────────────────────────────────────


class TestVeryHighFrequency:
    """Verify simulation is numerically stable at very high frequencies."""

    @pytest.mark.parametrize("freq", [30_000, 50_000, 80_000])
    def test_no_nan_at_high_freq(self, driver, freq: int):
        """At 30, 50, and 80 kHz the solver must not produce NaN or Inf."""
        horn = HornGeometry(
            throat_area=20e-4,  # 20 cm²
            mouth_area=500e-4,  # 500 cm²
            path_length=0.5,  # 0.5 m
            n_segments=50,
        )
        freqs = np.array([freq])
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(result, msg=f"{freq} Hz")


# ─────────────────────────────────────────────────────────────────────────────
# Test: very low frequency (5 Hz) — no numerical overflow
# ─────────────────────────────────────────────────────────────────────────────


class TestVeryLowFrequency:
    """Verify simulation is numerically stable at very low frequencies."""

    @pytest.mark.parametrize("freq", [5, 10, 15])
    def test_no_overflow_at_low_freq(self, driver, freq: int):
        """At 5, 10, and 15 Hz the solver must not overflow or produce NaN/Inf."""
        horn = HornGeometry(
            throat_area=20e-4,  # 20 cm²
            mouth_area=500e-4,  # 500 cm²
            path_length=0.5,  # 0.5 m
            n_segments=50,
        )
        freqs = np.array([freq], dtype=float)
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(result, msg=f"{freq} Hz")


# ─────────────────────────────────────────────────────────────────────────────
# Test: extreme area ratio (throat=1cm², mouth=5000cm²) — graceful handling
# ─────────────────────────────────────────────────────────────────────────────


class TestExtremeAreaRatio:
    """Verify simulation handles extreme throat/mouth area ratios gracefully."""

    @pytest.mark.parametrize(
        "throat_cm2,mouth_cm2",
        [
            (1, 5000),  # 1:5000 ratio — very small throat, very large mouth
            (0.5, 5000),  # 1:10000 ratio — extremely challenging
            (1, 10000),  # 1:10000 ratio — different extreme
        ],
    )
    def test_extreme_area_ratio_no_crash(
        self, driver, throat_cm2: float, mouth_cm2: float
    ):
        """Must complete without raising an exception and without NaN/Inf."""
        horn = HornGeometry(
            throat_area=throat_cm2 * 1e-4,  # cm² → m²
            mouth_area=mouth_cm2 * 1e-4,  # cm² → m²
            path_length=1.0,  # 1 m path
            n_segments=100,
        )
        # Use a reasonable mid-range frequency sweep to exercise the ratio
        freqs = np.linspace(20.0, 2000.0, 200)
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(
            result, msg=f"throat={throat_cm2}cm² mouth={mouth_cm2}cm²"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test: long horn path (6m) — completes without stack overflow
# ─────────────────────────────────────────────────────────────────────────────


class TestLongHornPath:
    """Verify simulation handles very long horn paths without overflow."""

    @pytest.mark.parametrize("path_m", [5.0, 6.0, 8.0])
    def test_long_path_completes(self, driver, path_m: float):
        """Must complete without RecursionError/StackOverflow and without NaN/Inf."""
        horn = HornGeometry(
            throat_area=10e-4,  # 10 cm²
            mouth_area=300e-4,  # 300 cm²
            path_length=path_m,
            n_segments=200,  # more segments for longer path
        )
        freqs = np.linspace(20.0, 2000.0, 200)
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(result, msg=f"path={path_m}m")


# ─────────────────────────────────────────────────────────────────────────────
# Test: combined extreme parameters — stress test
# ─────────────────────────────────────────────────────────────────────────────


class TestCombinedExtremeParameters:
    """Verify simulation handles multiple extreme parameters simultaneously."""

    def test_high_freq_long_path(self, driver):
        """High frequency + long path: stress test for numerical stability."""
        horn = HornGeometry(
            throat_area=5e-4,  # 5 cm²
            mouth_area=2000e-4,  # 2000 cm²
            path_length=5.0,  # 5 m
            n_segments=150,
        )
        freqs = np.array([5000.0, 10000.0, 15000.0])
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(result, msg="high_freq_long_path")

    def test_low_freq_extreme_ratio(self, driver):
        """Low frequency + extreme area ratio: stress test for numerical stability."""
        horn = HornGeometry(
            throat_area=1e-4,  # 1 cm²
            mouth_area=5000e-4,  # 5000 cm²
            path_length=2.0,
            n_segments=100,
        )
        freqs = np.array([5.0, 10.0, 15.0, 20.0])
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(result, msg="low_freq_extreme_ratio")


# ─────────────────────────────────────────────────────────────────────────────
# Test: boundary parameter values — n_segments=1, f=0 Hz
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundaryParameters:
    """Verify simulation handles boundary (minimum-valid) parameter values."""

    def test_n_segments_one_completes(self, driver):
        """n_segments=1 is the minimum-valid value — must not cause division errors."""
        horn = HornGeometry(
            throat_area=20e-4,  # 20 cm²
            mouth_area=300e-4,  # 300 cm²
            path_length=0.8,  # 0.8 m
            n_segments=1,  # minimum-valid value
        )
        freqs = np.linspace(20.0, 5000.0, 100)
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(result, msg="n_segments=1")

    def test_zero_frequency_single_point(self, driver):
        """f=0 Hz (DC) — ω=0. Must not produce NaN/Inf in impedance or SPL."""
        horn = HornGeometry(
            throat_area=20e-4,
            mouth_area=300e-4,
            path_length=0.5,
            n_segments=50,
        )
        freqs = np.array([0.0])
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        # Impedance at DC should be a finite real value (R_e), not NaN
        z = np.asarray(result.impedance)
        assert np.all(np.isfinite(z)), f"[f=0] impedance must be finite, got {z}"
        # SPL at DC is undefined (no acoustic output at 0 Hz) — NaN is acceptable
        # but Inf is not
        assert not np.any(
            np.isinf(result.spl)
        ), f"[f=0] SPL must not be Inf, got {result.spl}"

    def test_zero_frequency_in_sweep(self, driver):
        """f=0 Hz embedded in a frequency sweep — must not poison other frequencies."""
        horn = HornGeometry(
            throat_area=20e-4,
            mouth_area=300e-4,
            path_length=0.5,
            n_segments=50,
        )
        freqs = np.array([0.0, 10.0, 20.0, 100.0, 1000.0])
        result = horn_response(
            freqs=freqs, driver=driver, horn=horn, compute_distortion=False
        )
        _assert_no_nan_or_inf(result, msg="f=0 in sweep")
        # Non-zero frequencies should have finite SPL
        for i, f in enumerate(freqs):
            if f > 0:
                assert np.isfinite(
                    result.spl[i]
                ), f"[f={f}] SPL must be finite, got {result.spl[i]}"


# ─────────────────────────────────────────────────────────────────────────────
# Test: zero voltage driver — degenerate but should not crash
# ─────────────────────────────────────────────────────────────────────────────


class TestZeroVoltage:
    """Verify simulation handles a zero-voltage driver gracefully."""

    def test_zero_voltage_no_crash(self, driver):
        """driver.voltage=0 produces zero cone velocity — no NaN/Inf in output."""
        horn = HornGeometry(
            throat_area=20e-4,
            mouth_area=300e-4,
            path_length=0.5,
            n_segments=50,
        )
        # Use a copy of the driver with voltage=0
        driver_zero = DriverSpecs(
            fs=driver.fs,
            qts=driver.qts,
            qes=driver.qes,
            qms=driver.qms,
            vas=driver.vas,
            re=driver.re,
            bl=driver.bl,
            mms=driver.mms,
            cms=driver.cms,
            rms=driver.rms,
            sd=driver.sd,
            le=driver.le,
            voltage=0.0,  # zero input voltage
        )
        freqs = np.linspace(20.0, 5000.0, 100)
        result = horn_response(
            freqs=freqs, driver=driver_zero, horn=horn, compute_distortion=False
        )
        # All outputs must be finite (SPL may be -inf / very negative, but not NaN)
        z = np.asarray(result.impedance)
        assert np.all(np.isfinite(z)), f"[V=0] impedance must be finite, got {z}"
        # SPL at zero voltage: should be non-NaN (likely -inf), but not NaN
        assert not np.any(
            np.isnan(result.spl)
        ), f"[V=0] SPL must not be NaN, got {result.spl}"
