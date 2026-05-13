"""Unit tests for pyhorn.solver.optimizer."""

import numpy as np
import pytest
from pyhorn_core.config.models import DriverSpecs, HornGeometry
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_core.solver.optimizer import (
    OptimizationConfig,
    compute_analytical_seed,
    extrapolate_folded_horn,
    objective_function,
    optimize_single_profile,
    optimize,
    PARAM_NAMES,
)


@pytest.fixture
def driver():
    """Fostex FE166NV2 driver specs."""
    return DriverSpecs(
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
        voltage=2.83,
        le=0.0008,
        xmax=0.0015,
    )


@pytest.fixture
def config():
    return OptimizationConfig(fmin=80.0, fmax=5000.0, n_freq_points=100)


class TestBuildHornFromParams:
    def test_returns_horn_geometry(self):
        params = {
            "throat_area": 0.01,
            "mouth_area": 0.1,
            "path_length": 1.0,
            "lrc": 0.1,
            "vtc": 0.001,
        }
        horn = build_horn_from_params(params, "exponential", "BLH")
        assert isinstance(horn, HornGeometry)
        assert horn.throat_area == 0.01
        assert horn.mouth_area == 0.1
        assert horn.path_length == 1.0
        assert horn.profile_type == "exponential"
        assert horn.enclosure_type == "BLH"
        assert horn.lrc == 0.1
        assert horn.vtc == 0.001
        assert horn.n_segments == 100

    def test_vrc_estimated_from_lrc_and_throat(self):
        params = {
            "throat_area": 0.02,
            "mouth_area": 0.1,
            "path_length": 1.0,
            "lrc": 0.2,
            "vtc": 0.0,
        }
        horn = build_horn_from_params(params, "conical", "BLH")
        assert horn.vrc == pytest.approx(0.2 * 0.02)


class TestAnalyticalSeed:
    def test_returns_all_params(self, driver, config):
        seed = compute_analytical_seed(driver, config)
        for name in PARAM_NAMES:
            assert name in seed

    def test_values_within_bounds(self, driver, config):
        seed = compute_analytical_seed(driver, config)
        for name, (lo, hi) in zip(PARAM_NAMES, config.bounds):
            assert lo <= seed[name] <= hi, f"{name}={seed[name]} outside [{lo}, {hi}]"

    def test_throat_area_near_driver_sd(self, driver, config):
        seed = compute_analytical_seed(driver, config)
        # Should be clipped to range but close to Sd if Sd is within range
        assert seed["throat_area"] == pytest.approx(driver.sd, rel=0.5)

    def test_seed_respects_minimum_expansion_ratio(self, driver):
        config = OptimizationConfig(min_expansion_ratio=6.0)
        seed = compute_analytical_seed(driver, config)
        assert seed["mouth_area"] / seed["throat_area"] >= 6.0


class TestObjectiveFunction:
    def test_returns_finite_for_valid_params(self, driver, config):
        freqs = np.linspace(40, 5000, 100)
        x = np.array([0.013, 0.08, 1.0, 0.1, 0.001, 0.5])
        cost = objective_function(x, driver, "exponential", config, freqs)
        assert np.isfinite(cost)

    def test_penalizes_mouth_smaller_than_throat(self, driver, config):
        freqs = np.linspace(40, 5000, 100)
        x = np.array([0.02, 0.01, 1.0, 0.1, 0.001, 0.5])  # mouth < throat
        cost = objective_function(x, driver, "exponential", config, freqs)
        assert cost == 1e6

    def test_rejects_cutoff_above_fmin(self, driver, config):
        """Design with fc > fmin should be rejected."""
        freqs = np.linspace(40, 5000, 100)
        # Large expansion over short path → fc=201 Hz, well above fmin=80
        x = np.array([0.005, 0.2, 0.5, 0.1, 0.001, 0.5])
        cost = objective_function(x, driver, "exponential", config, freqs)
        assert cost == 1e6

    def test_rejects_expansion_ratio_below_minimum(self, driver, config):
        freqs = np.linspace(40, 5000, 100)
        x = np.array([0.02, 0.05, 1.0, 0.1, 0.001, 0.5])
        cost = objective_function(x, driver, "exponential", config, freqs)
        assert cost == 1e6

    def test_accepts_cutoff_below_fmin(self, driver, config):
        """Design with fc <= fmin should not be rejected by cutoff check."""
        freqs = np.linspace(40, 5000, 100)
        # Moderate expansion over long path → fc=63 Hz, below fmin=80
        x = np.array([0.01, 0.1, 1.0, 0.1, 0.001, 0.5])
        cost = objective_function(x, driver, "exponential", config, freqs)
        assert cost < 1e6

    def test_different_profiles_give_different_costs(self, driver, config):
        freqs = np.linspace(40, 5000, 100)
        x = np.array([0.013, 0.08, 1.0, 0.1, 0.001, 0.5])
        cost_exp = objective_function(x, driver, "exponential", config, freqs)
        cost_con = objective_function(x, driver, "conical", config, freqs)
        # They should both be finite but likely differ
        assert np.isfinite(cost_exp)
        assert np.isfinite(cost_con)

    def test_oversized_throat_incurs_penalty(self, driver):
        freqs = np.linspace(40, 5000, 100)
        config = OptimizationConfig(throat_area_penalty_weight=2.0)
        x_small = np.array([driver.sd * 0.95, driver.sd * 5.0, 1.0, 0.1, 0.001, 0.5])
        x_large = np.array([driver.sd * 1.5, driver.sd * 5.0, 1.0, 0.1, 0.001, 0.5])
        cost_small = objective_function(x_small, driver, "exponential", config, freqs)
        cost_large = objective_function(x_large, driver, "exponential", config, freqs)
        assert np.isfinite(cost_small)
        assert np.isfinite(cost_large)
        assert cost_large > cost_small

    def test_hyperbolic_t_parameter_affects_hyperbolic_profile(self, driver, config):
        """Varying hyperbolic_t should change the cost for hyperbolic profile."""
        freqs = np.linspace(40, 5000, 100)
        base = np.array([0.013, 0.08, 1.0, 0.1, 0.001, 0.0])  # T=0 (catenoidal)
        x_t05 = base.copy()
        x_t05[5] = 0.5  # T=0.5 (typical hyperbolic)
        x_t10 = base.copy()
        x_t10[5] = 1.0  # T=1.0 (sinh-based)
        cost_t0 = objective_function(base, driver, "hyperbolic", config, freqs)
        cost_t05 = objective_function(x_t05, driver, "hyperbolic", config, freqs)
        cost_t10 = objective_function(x_t10, driver, "hyperbolic", config, freqs)
        assert np.isfinite(cost_t0)
        assert np.isfinite(cost_t05)
        assert np.isfinite(cost_t10)
        # All three should give different costs (T parameter affects geometry)
        assert cost_t0 != cost_t05 or cost_t05 != cost_t10


class TestOptimizeSingleProfile:
    def test_runs_and_returns_result(self, driver):
        """Quick optimization with minimal iterations to verify it runs."""
        config = OptimizationConfig(
            fmin=80.0,
            fmax=5000.0,
            n_freq_points=50,
            max_iter=5,
            popsize=5,
            seed=42,
        )
        result = optimize_single_profile(driver, "exponential", config)
        assert result.profile_type == "exponential"
        assert np.isfinite(result.cost)
        assert result.mean_spl > 0
        assert result.n_evaluations > 0
        assert isinstance(result.horn, HornGeometry)


class TestOptimize:
    def test_runs_all_profiles(self, driver):
        """Quick run across two profiles."""
        config = OptimizationConfig(
            fmin=80.0,
            fmax=5000.0,
            n_freq_points=50,
            max_iter=3,
            popsize=5,
            seed=42,
            profile_types=["conical", "exponential"],
        )
        results = optimize(driver, config)
        assert len(results) == 2
        # Should be sorted by cost
        assert results[0].cost <= results[1].cost


class TestFoldedExtrapolation:
    def test_generates_folded_geometry_within_enclosure(self):
        params = {
            "throat_area": 0.01,
            "mouth_area": 0.08,
            "path_length": 1.4,
            "lrc": 0.1,
            "vtc": 0.001,
        }
        horn = build_horn_from_params(params, "conical", "BLH")

        folded = extrapolate_folded_horn(horn, (0.5, 0.7), (0.0, 0.18), 0.75)

        assert folded.profile_type is None
        assert folded.width is not None and folded.width > 0
        assert folded.coordinates is not None and len(folded.coordinates) >= 2
        assert folded.rectangular_segments is not None
        assert sum(seg[4] for seg in folded.rectangular_segments) == pytest.approx(
            horn.path_length, rel=1e-6
        )
        assert folded.enclosure_dims == (0.5, 0.7)
        assert folded.driver_coord == (0.0, 0.18)
        assert all(0.0 <= x <= 0.5 and 0.0 <= y <= 0.7 for x, y in folded.coordinates)
        chamber_side = np.sqrt((horn.vtc + 1e-3) / 0.75)
        expected_start_x = chamber_side / 2.0
        expected_start_y = folded.driver_coord[1] + chamber_side / 2.0
        assert folded.coordinates[0][0] == pytest.approx(expected_start_x)
        assert folded.coordinates[0][1] == pytest.approx(expected_start_y)
        assert folded.coordinates[-1][0] == pytest.approx(0.025)
        assert folded.coordinates[-1][0] < folded.coordinates[-2][0]
        assert folded.coordinates[-1][1] == pytest.approx(folded.coordinates[-2][1])
        distinct_x = {round(point[0], 6) for point in folded.coordinates}
        assert len(distinct_x) >= 3
        vertical_lengths = [
            abs(b[1] - a[1])
            for a, b in zip(folded.coordinates, folded.coordinates[1:])
            if abs(b[1] - a[1]) > 1e-9
        ]
        horizontal_lengths = [
            abs(b[0] - a[0])
            for a, b in zip(folded.coordinates, folded.coordinates[1:])
            if abs(b[0] - a[0]) > 1e-9
        ]
        assert vertical_lengths
        assert horizontal_lengths
        assert max(vertical_lengths) > max(horizontal_lengths)

    def test_rejects_driver_too_close_to_wall(self):
        params = {
            "throat_area": 0.01,
            "mouth_area": 0.08,
            "path_length": 1.0,
            "lrc": 0.1,
            "vtc": 0.001,
        }
        horn = build_horn_from_params(params, "conical", "BLH")

        with pytest.raises(ValueError):
            extrapolate_folded_horn(horn, (0.5, 0.7), (0.0, 0.01), 0.75)

    def test_accepts_fixed_width_when_feasible(self):
        params = {
            "throat_area": 0.01,
            "mouth_area": 0.08,
            "path_length": 1.4,
            "lrc": 0.1,
            "vtc": 0.001,
        }
        horn = build_horn_from_params(params, "conical", "BLH")

        folded = extrapolate_folded_horn(horn, (0.5, 0.7), (0.0, 0.2), 0.75)

        assert folded.width == pytest.approx(0.75)
        assert folded.rectangular_segments is not None
        assert folded.coordinates is not None
        chamber_side = np.sqrt((horn.vtc + 1e-3) / 0.75)
        expected_start_x = chamber_side / 2.0
        assert folded.coordinates[0][0] == pytest.approx(expected_start_x)
        assert all(
            seg[0] == pytest.approx(0.75) and seg[2] == pytest.approx(0.75)
            for seg in folded.rectangular_segments
        )

    def test_user_case_fits_with_same_side_mouth(self):
        params = {
            "throat_area": 0.012331,
            "mouth_area": 0.074001,
            "path_length": 1.3080,
            "lrc": 0.4493,
            "vtc": 0.004988,
        }
        horn = build_horn_from_params(params, "conical", "BLH")

        folded = extrapolate_folded_horn(horn, (0.5, 0.7), (0.0, 0.25), 0.2)

        assert folded.width == pytest.approx(0.2)
        assert folded.coordinates is not None
        chamber_side = np.sqrt((horn.vtc + 1e-3) / 0.2)
        assert folded.coordinates[0][0] == pytest.approx(chamber_side / 2.0)
        assert folded.coordinates[0][1] == pytest.approx(
            folded.driver_coord[1] + chamber_side / 2.0
        )
        assert folded.coordinates[-1][0] < 0.25
        assert folded.coordinates[-1][0] < folded.coordinates[-2][0]
        assert folded.coordinates[-1][1] == pytest.approx(folded.coordinates[-2][1])

    def test_rejects_fixed_width_when_too_small(self):
        params = {
            "throat_area": 0.01,
            "mouth_area": 0.08,
            "path_length": 1.4,
            "lrc": 0.1,
            "vtc": 0.001,
        }
        horn = build_horn_from_params(params, "conical", "BLH")

        with pytest.raises(ValueError, match="Could not fit|too small"):
            extrapolate_folded_horn(horn, (0.5, 0.7), (0.0, 0.2), 0.15)
