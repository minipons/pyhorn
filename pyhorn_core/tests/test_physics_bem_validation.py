"""
Tests for pyhorn_physics.bem_validation module.

Tests cover:
  - BEM reference data loading (CSV, JSON)
  - TMM prediction creation
  - SPL comparison (delta, RMSE, max delta)
  - Impedance comparison
  - Threshold assessment
  - Standard geometry generation
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyhorn_core.pyhorn_physics.bem_validation import (
    BemReferenceData,
    BemValidationResult,
    TmmPredictionData,
    load_bem_reference,
    interpolate_to_reference,
    compare_spl,
    compare_impedance,
    assess_tmm_validity,
    generate_standard_horn_reference,
)


REPO = Path(__file__).resolve().parents[2]
BEM_BENCHMARK = REPO / "tests/benchmarks/bem/exponential_horn"
REFERENCE_CSV = BEM_BENCHMARK / "reference/spl.csv"
FIXTURE_HORN = BEM_BENCHMARK / "fixture/horn.yaml"


class TestLoadBemReference:
    """Loading BEM reference data from CSV/JSON files."""

    def test_load_csv_exists(self):
        """Load a CSV reference file that exists."""
        ref = load_bem_reference(REFERENCE_CSV)
        assert ref.freqs is not None
        assert len(ref.freqs) > 0
        assert ref.has_spl()

    def test_load_csv_creates_bem_reference_data(self):
        """Returned object is a BemReferenceData instance."""
        ref = load_bem_reference(REFERENCE_CSV)
        assert isinstance(ref, BemReferenceData)

    def test_load_csv_freqs_match_header(self):
        """Loaded frequencies match CSV column."""
        ref = load_bem_reference(REFERENCE_CSV)
        assert len(ref.freqs) > 20

    def test_load_csv_missing_file_raises(self):
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_bem_reference("/nonexistent/path/to/bem_reference.csv")

    def test_load_bem_csv_impedance_columns(self):
        """CSV with Z_real/Z_imag columns parses correctly."""
        ref = load_bem_reference(REFERENCE_CSV)
        assert ref.impedance_real is not None
        assert ref.impedance_imag is not None
        assert ref.has_impedance()


class TestInterpolateToReference:
    """Log-frequency interpolation for mismatched frequency grids."""

    def test_interpolate_same_grid(self):
        """Same frequency grid returns identical values."""
        freqs = np.linspace(20, 1000, 50)
        values = np.sin(2 * np.pi * freqs / 200)
        result = interpolate_to_reference(freqs, freqs, values)
        np.testing.assert_allclose(result, values, atol=1e-10)

    def test_interpolate_coarse_to_fine(self):
        """Coarse grid interpolated to fine grid is smooth."""
        source_freqs = np.array([20.0, 100.0, 1000.0])
        source_vals = np.array([0.0, 1.0, 0.5])
        target_freqs = np.logspace(np.log10(20), np.log10(1000), 100)
        result = interpolate_to_reference(target_freqs, source_freqs, source_vals)
        assert len(result) == len(target_freqs)
        assert np.all(np.isfinite(result))

    def test_interpolate_extrapolation_nan(self):
        """Frequencies outside source range return NaN."""
        source_freqs = np.array([100.0, 1000.0])
        source_vals = np.array([1.0, 2.0])
        target_freqs = np.array([50.0, 5000.0])
        result = interpolate_to_reference(target_freqs, source_freqs, source_vals)
        assert np.isnan(result[0])
        assert np.isnan(result[-1])

    def test_interpolate_empty_source(self):
        """Empty source array returns NaN array."""
        result = interpolate_to_reference(
            np.array([100.0, 200.0]),
            np.array([]),
            np.array([]),
        )
        assert all(np.isnan(result))


class TestCompareSpl:
    """SPL comparison between BEM reference and TMM prediction."""

    def test_compare_identical_data_zero_delta(self):
        """Identical BEM and TMM data gives zero delta."""
        freqs = np.linspace(20, 5000, 50)
        spl = 90.0 + 5.0 * np.sin(2 * np.pi * freqs / 500)

        bem = BemReferenceData(freqs=freqs, spl=spl)
        tmm = TmmPredictionData(freqs=freqs, spl=spl)

        result = compare_spl(bem, tmm)
        np.testing.assert_allclose(result.delta_spl, 0.0, atol=1e-9)
        assert result.rmse_db == 0.0
        assert result.max_abs_delta_db == 0.0

    def test_compare_small_difference_small_rmse(self):
        """Small 0.5 dB uniform offset gives RMSE = 0.5 dB."""
        freqs = np.linspace(20, 5000, 100)
        bem_spl = 90.0 * np.ones_like(freqs)
        tmm_spl = bem_spl + 0.5

        bem = BemReferenceData(freqs=freqs, spl=bem_spl)
        tmm = TmmPredictionData(freqs=freqs, spl=tmm_spl)

        result = compare_spl(bem, tmm)
        np.testing.assert_allclose(result.rmse_db, 0.5, rtol=1e-4)
        np.testing.assert_allclose(result.mean_delta_db, 0.5, rtol=1e-4)

    def test_compare_passes_within_threshold(self):
        """Result pass_threshold is True when RMSE < threshold."""
        freqs = np.linspace(20, 5000, 100)
        bem = BemReferenceData(freqs=freqs, spl=90.0 * np.ones_like(freqs))
        tmm = TmmPredictionData(freqs=freqs, spl=91.0 * np.ones_like(freqs))

        result = compare_spl(bem, tmm, threshold_db=2.0)
        assert result.pass_threshold is True

    def test_compare_fails_outside_threshold(self):
        """Result pass_threshold is False when RMSE > threshold."""
        freqs = np.linspace(20, 5000, 100)
        bem = BemReferenceData(freqs=freqs, spl=90.0 * np.ones_like(freqs))
        tmm = TmmPredictionData(freqs=freqs, spl=100.0 * np.ones_like(freqs))

        result = compare_spl(bem, tmm, threshold_db=3.0)
        assert result.pass_threshold is False

    def test_compare_with_mismatched_frequencies(self):
        """Handles mismatched frequency grids via interpolation."""
        bem_freqs = np.linspace(20, 5000, 50)
        tmm_freqs = np.logspace(np.log10(20), np.log10(5000), 100)
        bem_spl = 90.0 + 0.1 * np.sin(2 * np.pi * bem_freqs / 500)
        tmm_spl = 90.0 + 0.1 * np.sin(2 * np.pi * tmm_freqs / 500)

        bem = BemReferenceData(freqs=bem_freqs, spl=bem_spl)
        tmm = TmmPredictionData(freqs=tmm_freqs, spl=tmm_spl)

        result = compare_spl(bem, tmm)
        assert result.rmse_db < 0.5

    def test_compare_no_spl_raises(self):
        """Missing SPL data raises ValueError."""
        bem = BemReferenceData(freqs=np.array([100.0]))
        tmm = TmmPredictionData(freqs=np.array([100.0]), spl=np.array([90.0]))

        with pytest.raises(ValueError, match="no SPL"):
            compare_spl(bem, tmm)

    def test_compare_insufficient_overlap_raises(self):
        """Insufficient frequency overlap raises ValueError."""
        bem = BemReferenceData(freqs=np.array([20.0, 30.0, 40.0]), spl=np.array([80.0, 85.0, 90.0]))
        tmm = TmmPredictionData(freqs=np.array([5000.0, 6000.0, 7000.0]), spl=np.array([90.0, 95.0, 100.0]))

        with pytest.raises(ValueError, match="overlapping"):
            compare_spl(bem, tmm)


class TestCompareImpedance:
    """Impedance comparison between BEM and TMM."""

    def test_compare_impedance_identical(self):
        """Identical impedance gives zero delta and zero RMSE."""
        freqs = np.linspace(20, 5000, 50)
        z = 7.0 + 1j * np.linspace(0, 10, 50)

        bem = BemReferenceData(
            freqs=freqs,
            impedance_real=z.real,
            impedance_imag=z.imag,
        )
        tmm = TmmPredictionData(freqs=freqs, spl=np.ones_like(freqs) * 90.0, impedance=z)

        delta_r, delta_i, rmse = compare_impedance(bem, tmm)
        np.testing.assert_allclose(delta_r, 0.0, atol=1e-9)
        np.testing.assert_allclose(delta_i, 0.0, atol=1e-9)
        assert rmse == 0.0


class TestAssessTmmValidity:
    """Assessment of TMM validity based on comparison results."""

    def test_assess_all_pass(self):
        """All metrics within thresholds → overall_pass = True."""
        result = BemValidationResult(
            freqs=np.linspace(20, 5000, 100),
            delta_spl=np.random.randn(100) * 0.5,
            mean_delta_db=0.5,
            std_delta_db=0.5,
            rmse_db=1.5,
            max_abs_delta_db=3.0,
            max_delta_db=3.0,
            min_delta_db=-3.0,
        )

        assessment = assess_tmm_validity(result)
        assert assessment["overall_pass"] is True

    def test_assess_rmse_fail(self):
        """RMSE above threshold → overall_pass = False."""
        result = BemValidationResult(
            freqs=np.linspace(20, 5000, 100),
            delta_spl=np.ones(100) * 5.0,
            mean_delta_db=5.0,
            std_delta_db=0.1,
            rmse_db=5.0,
            max_abs_delta_db=5.0,
            max_delta_db=5.0,
            min_delta_db=5.0,
        )

        assessment = assess_tmm_validity(
            result, thresholds={"rmse_db": 3.0, "max_abs_db": 10.0, "mean_db": 5.0, "std_db": 5.0}
        )
        assert assessment["overall_pass"] is False
        assert assessment["checks"]["rmse"] is False


class TestGenerateStandardHornReference:
    """Standard horn geometry generation for BEM validation."""

    def test_generate_exponential_horn(self):
        """Exponential horn geometry generation succeeds."""
        geo = generate_standard_horn_reference(
            geometry_type="exponential",
            throat_area_m2=0.0005,
            mouth_area_m2=0.01,
            path_length_m=0.3,
        )

        assert geo["geometry_type"] == "exponential"
        assert geo["throat_area_m2"] == 0.0005
        assert geo["mouth_area_m2"] == 0.01
        assert "profile_params" in geo
        assert "m" in geo["profile_params"]

    def test_generate_conical_horn(self):
        """Conical horn geometry generation succeeds."""
        geo = generate_standard_horn_reference(
            geometry_type="conical",
            throat_area_m2=0.0005,
            mouth_area_m2=0.01,
            path_length_m=0.3,
        )

        assert geo["geometry_type"] == "conical"
        assert "throat_radius" in geo["profile_params"]
        assert "mouth_radius" in geo["profile_params"]

    def test_generate_hyperbolic_horn(self):
        """Hyperbolic horn geometry generation succeeds."""
        geo = generate_standard_horn_reference(
            geometry_type="hyperbolic",
            throat_area_m2=0.0005,
            mouth_area_m2=0.01,
            path_length_m=0.3,
        )

        assert geo["geometry_type"] == "hyperbolic"
        assert "t" in geo["profile_params"]

    def test_generate_invalid_area_raises(self):
        """Zero or negative area raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            generate_standard_horn_reference(
                geometry_type="exponential",
                throat_area_m2=0.0,
                mouth_area_m2=0.01,
                path_length_m=0.3,
            )


class TestBemValidationResult:
    """BemValidationResult dataclass behavior."""

    def test_has_impedance_property(self):
        """impedance property returns complex array from real/imag."""
        bem = BemReferenceData(
            freqs=np.array([100.0, 200.0]),
            impedance_real=np.array([7.0, 8.0]),
            impedance_imag=np.array([1.0, 2.0]),
        )

        z = bem.impedance
        assert z is not None
        np.testing.assert_array_equal(z.real, np.array([7.0, 8.0]))
        np.testing.assert_array_equal(z.imag, np.array([1.0, 2.0]))

    def test_has_spl(self):
        """has_spl returns True when SPL data is present."""
        bem = BemReferenceData(freqs=np.array([100.0]), spl=np.array([90.0]))
        assert bem.has_spl() is True

        bem_empty = BemReferenceData(freqs=np.array([100.0]), spl=None)
        assert bem_empty.has_spl() is False

    def test_has_impedance(self):
        """has_impedance returns True when impedance data is present."""
        bem = BemReferenceData(
            freqs=np.array([100.0]),
            impedance_real=np.array([7.0]),
            impedance_imag=np.array([1.0]),
        )
        assert bem.has_impedance() is True

    def test_has_directivity(self):
        """has_directivity returns True when directivity data is present."""
        angles = np.array([0.0, 15.0, 30.0])
        bem = BemReferenceData(
            freqs=np.array([100.0]),
            directivity_horizontal=np.array([0.0, -2.0, -5.0]),
            directivity_angles=angles,
        )
        assert bem.has_directivity() is True
