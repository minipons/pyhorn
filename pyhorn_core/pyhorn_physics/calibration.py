from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d as _scipy_interp1d


def _pressure_to_spl(pressure: np.ndarray) -> np.ndarray:
    return 20 * np.log10(np.abs(pressure) / 2e-5 + 1e-12)


def _power_sum_spl(*spl_components: np.ndarray | None) -> np.ndarray:
    total_power = None
    for component in spl_components:
        if component is None:
            continue
        component_power = np.power(10.0, np.asarray(component, dtype=float) / 10.0)
        total_power = (
            component_power if total_power is None else total_power + component_power
        )

    if total_power is None:
        return np.array([], dtype=float)

    return 10.0 * np.log10(np.maximum(total_power, 1e-12))


def _interp1d_scalar(x, y, x_out):
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    y_arr = np.atleast_1d(np.asarray(y, dtype=float))
    f = _scipy_interp1d(
        x_arr,
        y_arr,
        kind="linear",
        fill_value="extrapolate",
        assume_sorted=True,
    )
    result = f(np.atleast_1d(np.asarray(x_out, dtype=float)))
    if np.ndim(result) == 1 and result.shape[0] == 1:
        return result[0]
    return result


def acoustic_power_to_spl_dB_W_m(
    p_acoustic_W: float | np.ndarray,
    sensitivity_db: float = 0.0,
) -> float | np.ndarray:
    p_ref = 1e-12
    with np.errstate(divide="ignore", invalid="ignore"):
        spl = 10.0 * np.log10(np.maximum(np.asarray(p_acoustic_W), p_ref) / p_ref)
        spl += sensitivity_db
    return spl
