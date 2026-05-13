"""
Horn geometry optimizer — finds optimal horn parameters for a given driver.

Uses scipy differential_evolution to search over 6 continuous parameters
(throat area, mouth area, path length, rear chamber length, throat chamber
volume, hyperbolic T) independently for each flare profile type, then ranks results.

The hyperbolic T parameter controls the shape of the hyperbolic flare family:
  T = 0.0  → catenoidal (T→0 limit)
  T = 0.5  → typical hyperbolic (default for hyperbolic profile search)
  T = 1.0  → sinh-based expansion
  T > 1.0  → increasingly extreme (faster initial flare, gentler toward mouth)

Hard constraint: the exponential-equivalent cutoff frequency must be at or
below fmin, ensuring the horn loads the driver down to the target frequency.

Objective (minimised):
    cost = w_flat * std(SPL)             # penalise ripple in [fmin, fmax]
         - w_sens * mean(SPL) / 100      # reward sensitivity
         + w_bass * bass_deficit / 12    # penalise LF rolloff vs mid-band
         + w_exc  * (exc_violation)^2    # penalise excursion beyond xmax
         + w_throat * max(0, St/Sd - 1)^2  # penalise oversized throats
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Callable
from scipy.optimize import differential_evolution

from pyhorn_core.config.driver_models import DriverSpecs
from pyhorn_core.config.horn_models import HornGeometry
from pyhorn_core.config.design_space import (
    ALL_PROFILE_TYPES,
    MOUTH_AREA_RANGE,
    OPTIMIZER_LRC_RANGE,
    OPTIMIZER_VTC_RANGE,
    PATH_LENGTH_RANGE,
    THROAT_AREA_RANGE,
)
from pyhorn_core.solver.design import build_horn_from_params
from pyhorn_core.pyhorn_physics import C
from pyhorn_core.pyhorn_physics.orchestrators import horn_response  # noqa: N811
from pyhorn_core.solver.scoring import compute_response_metrics
from pyhorn_core.solver.profiles import profile_area_at_distance  # noqa: F401

PROFILE_TYPES = list(ALL_PROFILE_TYPES)

# Order of the 6 continuous parameters in the DE vector.
PARAM_NAMES = ["throat_area", "mouth_area", "path_length", "lrc", "vtc", "hyperbolic_t"]


# ─── Configuration & result dataclasses ──────────────────────────────────────


@dataclass
class OptimizationConfig:
    """All knobs for the optimizer: target band, weights, search bounds, DE settings."""

    # Target frequency band (Hz)
    fmin: float = 80.0
    fmax: float = 5000.0
    n_freq_points: int = 200

    # Objective weights (see module docstring for the cost formula)
    flatness_weight: float = 1.0
    sensitivity_weight: float = 0.3
    bass_extension_weight: float = 0.5
    excursion_penalty_weight: float = 2.0
    throat_area_penalty_weight: float = 0.5

    # Horn-likeness constraint
    min_expansion_ratio: float = 4.0

    # Search bounds — (min, max) in SI units
    throat_area_range: Tuple[float, float] = THROAT_AREA_RANGE  # m²
    mouth_area_range: Tuple[float, float] = MOUTH_AREA_RANGE  # m²
    path_length_range: Tuple[float, float] = PATH_LENGTH_RANGE  # m
    lrc_range: Tuple[float, float] = OPTIMIZER_LRC_RANGE  # m
    vtc_range: Tuple[float, float] = OPTIMIZER_VTC_RANGE  # m³
    # Hyperbolic T parameter: 0=catenoidal, 0.5=typical hyperbolic,
    # 1=sinh-based, >1=more extreme. Only meaningful for hyperbolic profile.
    hyperbolic_t_range: Tuple[float, float] = (0.0, 2.0)

    enclosure_type: str = "BLH"
    profile_types: Optional[List[str]] = None  # None → all four

    # scipy.optimize.differential_evolution settings
    max_iter: int = 150
    popsize: int = 15
    tol: float = 1e-3
    seed: Optional[int] = None

    top_n: int = 3

    @property
    def bounds(self) -> List[Tuple[float, float]]:
        """DE bounds, one (min, max) pair per parameter in PARAM_NAMES order."""
        return [
            self.throat_area_range,
            self.mouth_area_range,
            self.path_length_range,
            self.lrc_range,
            self.vtc_range,
            self.hyperbolic_t_range,
        ]

    @property
    def profiles(self) -> List[str]:
        return self.profile_types or PROFILE_TYPES


@dataclass
class OptimizationResult:
    """Outcome of optimising one profile type."""

    profile_type: str
    params: dict  # keys = PARAM_NAMES, values in SI
    cost: float  # raw objective (lower = better)
    flatness_db: float  # std dev of SPL in band
    mean_spl: float  # mean SPL in band (dB)
    bass_deficit_db: float  # SPL drop in [fmin, 2*fmin] vs mid-band
    excursion_ok: bool  # True if peak excursion <= xmax
    horn: HornGeometry  # ready-to-simulate geometry
    n_evaluations: int  # total solver calls


# ─── Helpers ─────────────────────────────────────────────────────────────────


def compute_analytical_seed(
    driver: DriverSpecs,
    config: OptimizationConfig,
) -> dict:
    """Physics-based starting point injected into the DE population.

    - path_length from quarter-wave tuning at fmin
    - throat_area matched to driver Sd
    - mouth_area from exponential cutoff formula at fmin
    """
    fmin = config.fmin

    path_length = float(np.clip(C / (4 * fmin), *config.path_length_range))
    throat_area = float(np.clip(driver.sd, *config.throat_area_range))

    m = 4 * np.pi * fmin / C
    mouth_area = float(
        np.clip(throat_area * np.exp(m * path_length), *config.mouth_area_range)
    )
    mouth_area = max(mouth_area, throat_area * config.min_expansion_ratio)
    mouth_area = float(np.clip(mouth_area, *config.mouth_area_range))

    return {
        "throat_area": throat_area,
        "mouth_area": mouth_area,
        "path_length": path_length,
        "lrc": float(np.clip(0.1, *config.lrc_range)),
        "vtc": float(np.clip(0.001, *config.vtc_range)),
        "hyperbolic_t": float(np.clip(0.5, *config.hyperbolic_t_range)),
    }


# ─── Objective function ──────────────────────────────────────────────────────


def objective_function(
    x: np.ndarray,
    driver: DriverSpecs,
    profile_type: str,
    config: OptimizationConfig,
    freqs: np.ndarray,
) -> float:
    """Cost function for differential_evolution (lower = better).

    Returns 1e6 for infeasible designs: mouth <= throat, cutoff > fmin,
    or solver failure.
    """
    params = dict(zip(PARAM_NAMES, x))

    if params["mouth_area"] <= params["throat_area"]:
        return 1e6

    expansion_ratio = params["mouth_area"] / params["throat_area"]
    if expansion_ratio < config.min_expansion_ratio:
        return 1e6

    # Hard constraint: exponential-equivalent cutoff must be <= fmin
    m = np.log(params["mouth_area"] / params["throat_area"]) / params["path_length"]
    fc = (m * C) / (4 * np.pi)
    if fc > config.fmin:
        return 1e6

    horn = build_horn_from_params(params, profile_type, config.enclosure_type)

    try:
        result = horn_response(freqs, driver, horn)
    except Exception:
        return 1e6

    spl = result.spl
    if spl is None or len(spl) == 0:
        return 1e6

    metrics = compute_response_metrics(spl, freqs, config.fmin, config.fmax)
    band_spl = spl[(freqs >= config.fmin) & (freqs <= config.fmax)]
    if len(band_spl) < 2:
        return 1e6

    # Flatness: std dev of SPL in band (dB)
    flatness_cost = metrics.flatness_db

    # Sensitivity: mean SPL in band (higher = better → negate)
    mean_spl = metrics.mean_spl
    sensitivity_cost = -mean_spl / 100.0

    # Bass extension: SPL deficit in [fmin, 2*fmin] relative to mid-band
    bass_cost = (
        metrics.bass_deficit_db / 12.0 if metrics.bass_mean_spl is not None else 1.0
    )

    # Excursion penalty: quadratic above xmax
    exc_cost = 0.0
    if driver.xmax > 0 and result.excursion is not None:
        xmax_mm = driver.xmax * 1000.0
        violation = np.max(result.excursion) - xmax_mm
        if violation > 0:
            exc_cost = (violation / xmax_mm) ** 2

    throat_cost = 0.0
    if driver.sd > 0:
        throat_ratio = params["throat_area"] / driver.sd
        if throat_ratio > 1.0:
            throat_cost = (throat_ratio - 1.0) ** 2

    return float(
        config.flatness_weight * flatness_cost
        + config.sensitivity_weight * sensitivity_cost
        + config.bass_extension_weight * bass_cost
        + config.excursion_penalty_weight * exc_cost
        + config.throat_area_penalty_weight * throat_cost
    )


# ─── Optimisation loop ──────────────────────────────────────────────────────


def optimize_single_profile(
    driver: DriverSpecs,
    profile_type: str,
    config: OptimizationConfig,
    callback: Optional[Callable] = None,
) -> OptimizationResult:
    """Run differential_evolution for one profile type and return scored result."""
    freqs = np.linspace(config.fmin * 0.5, config.fmax, config.n_freq_points)

    seed_x = np.array(
        [compute_analytical_seed(driver, config)[name] for name in PARAM_NAMES]
    )

    n_eval = [0]

    def _objective(x):
        n_eval[0] += 1
        return objective_function(x, driver, profile_type, config, freqs)

    de_result = differential_evolution(
        _objective,
        bounds=config.bounds,
        x0=seed_x,
        maxiter=config.max_iter,
        popsize=config.popsize,
        tol=config.tol,
        rng=config.seed,
        callback=callback,
        disp=False,
    )

    best_params = dict(zip(PARAM_NAMES, de_result.x))
    horn = build_horn_from_params(best_params, profile_type, config.enclosure_type)

    # Re-evaluate the winner for detailed scoring
    sim = horn_response(freqs, driver, horn)
    metrics = compute_response_metrics(sim.spl, freqs, config.fmin, config.fmax)

    exc_ok = True
    if driver.xmax > 0 and sim.excursion is not None:
        exc_ok = float(np.max(sim.excursion)) <= driver.xmax * 1000.0

    return OptimizationResult(
        profile_type=profile_type,
        params=best_params,
        cost=float(de_result.fun),
        flatness_db=metrics.flatness_db,
        mean_spl=metrics.mean_spl,
        bass_deficit_db=metrics.bass_deficit_db,
        excursion_ok=exc_ok,
        horn=horn,
        n_evaluations=n_eval[0],
    )


def optimize(
    driver: DriverSpecs,
    config: OptimizationConfig,
    progress_callback: Optional[Callable] = None,
) -> List[OptimizationResult]:
    """Optimise across all selected profile types, return results sorted by cost."""
    results = []
    for profile in config.profiles:
        if progress_callback:
            progress_callback(f"Optimizing {profile} profile...")
        result = optimize_single_profile(driver, profile, config)
        results.append(result)
        if progress_callback:
            progress_callback(
                f"  {profile}: cost={result.cost:.3f}, "
                f"SPL={result.mean_spl:.1f} dB, "
                f"ripple={result.flatness_db:.1f} dB, "
                f"bass deficit={result.bass_deficit_db:.1f} dB"
            )

    results.sort(key=lambda r: r.cost)
    return results
