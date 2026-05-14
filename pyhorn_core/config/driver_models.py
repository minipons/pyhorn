from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class DriverSpecs:
    """Thiele-Small parameters for the loudspeaker driver."""

    fs: float
    qts: float
    qes: float
    qms: float
    vas: float
    re: float
    bl: float
    mms: float
    cms: float
    rms: float
    sd: float
    voltage: float = 2.83
    le: float = 0.0
    xmax: float = 0.0
    alpha_re: float = 0.00393
    le_freq_dependency: bool = False
    le_f_ref: float = 100.0
    lossy_le: bool = False
    le_R_e_eddy: float = 0.0
    le_f_lossy_ref: float = 1000.0
    sensitivity_db: float = 0.0
    spl_response: Optional[Any] = None

    def get_sensitivity_db(self, freqs: np.ndarray) -> Any:
        sd = self.sensitivity_db
        if isinstance(sd, np.ndarray):
            if sd.ndim == 2 and sd.shape[1] == 2:
                table_freqs = sd[:, 0]
                table_vals = sd[:, 1]
                return np.interp(
                    freqs,
                    table_freqs,
                    table_vals,
                    left=table_vals[0],
                    right=table_vals[-1],
                )
            return np.asarray(sd)
        return np.full_like(freqs, sd, dtype=float)

    def get_spl_response(self, freqs: np.ndarray) -> Optional[Any]:
        sr = self.spl_response
        if sr is None:
            return None
        arr = np.asarray(sr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        table_f = arr[:, 0]
        table_db = arr[:, 1]
        log_f = np.log10(np.maximum(freqs, 1e-6))
        log_tf = np.log10(np.maximum(table_f, 1e-6))

        try:
            from scipy.interpolate import PchipInterpolator

            interp = PchipInterpolator(log_tf, table_db, extrapolate=False)
            result = interp(log_f)
            result[log_f < log_tf[0]] = table_db[0]
            result[log_f > log_tf[-1]] = table_db[-1]
            return result
        except ImportError:
            return np.interp(
                log_f, log_tf, table_db, left=table_db[0], right=table_db[-1]
            )

    @property
    def reference_spl(self) -> float:
        rho = 1.21
        c = 343.0
        eta_0 = (rho / (2 * math.pi * c)) * (
            (self.bl**2 * self.sd**2) / (self.re * self.mms**2)
        )
        spl_1w = 112.2 + 10 * math.log10(max(eta_0, 1e-12))
        power = (self.voltage**2) / self.re
        return spl_1w + 10 * math.log10(max(power, 1e-12))
