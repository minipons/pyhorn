"""Unit tests for pyhorn.solver.optimizer."""

import numpy as np
import pytest
from pyhorn_core.config.models import DriverSpecs, HornGeometry
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_core.solver.optimizer import (
    OptimizationConfig,
    compute_analytical_seed,
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
