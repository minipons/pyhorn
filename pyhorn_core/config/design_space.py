"""Shared design-space defaults for horn optimization layers."""

from typing import Final, Tuple


THROAT_AREA_RANGE: Final[Tuple[float, float]] = (0.005, 0.03)
MOUTH_AREA_RANGE: Final[Tuple[float, float]] = (0.02, 0.2)
PATH_LENGTH_RANGE: Final[Tuple[float, float]] = (0.3, 2.0)

# Differential-evolution defaults used by pyhorn's physics-first optimizer.
OPTIMIZER_LRC_RANGE: Final[Tuple[float, float]] = (0.01, 0.5)
OPTIMIZER_VTC_RANGE: Final[Tuple[float, float]] = (0.0, 0.005)

# Bayesian defaults used by pyhorn_ml's exploratory search.
ML_LRC_RANGE: Final[Tuple[float, float]] = (0.03, 0.5)
ML_VTC_RANGE: Final[Tuple[float, float]] = (0.002, 0.005)

# All supported flare profiles (physics optimizer uses all four).
ALL_PROFILE_TYPES: Final[Tuple[str, ...]] = (
    "conical",
    "exponential",
    "hyperbolic",
    "parabolic",
)

# ML search omits parabolic — it rarely wins on the Pareto front.
ML_PROFILE_TYPES: Final[Tuple[str, ...]] = (
    "conical",
    "exponential",
    "hyperbolic",
)

FOLD_STYLES: Final[Tuple[str, ...]] = ("straight", "W", "pi")
N_SEGMENTS_RANGE: Final[Tuple[int, int]] = (10, 60)
