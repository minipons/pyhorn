"""Horn Segment Wizard — catenoidal horn geometry calculator.

Provides the core geometry formulas for a single catenoidal (T=1) horn segment.
Given any 3 of (S1 throat area, S2 mouth area, L12 horn length, F12 cutoff freq),
computes the 4th using the catenoidal horn formulas.

Reference: Hornresp manual page 63 (Horn Segment Wizard).

Units convention
----------------
Public API uses SI (m², m, Hz).  The CLI wrapper handles cm²/cm → m²/m conversion.

Formulas (catenoidal, T=1)
--------------------------
  F12  = c/(2π) × √(S2/S1 − 1) / L12
  L12  = c/(2π × F12) × √(S2/S1 − 1)
  S2   = S1 / (1 + (2π×F12×L12/c)²)
  S1   = S2 / (1 + (2π×F12×L12/c)²)     ← same structure, S1↔S2 swapped

Area profile (20 points including throat and mouth):
  A(x) = S1 × coth²(m·x),  m·L = arccosh(√(S2/S1))

System volume:
  V_horn = S1/(2m) · [sinh(u_total)·cosh(u_total) − u_total]
  u_total = arccosh(√(S2/S1)),  m = u_total/L
  V_system = V_horn + 0.1 L  (throat chamber estimate)

Usage
-----
>>> from pyhorn_core.solver.horn_segment import compute_horn_segment, HornSegmentResult
>>> result = compute_horn_segment(s1_m2=40e-4, s2_m2=None, l12_m=1.5, f12_hz=50.0)
>>> print(result.computed_param, result.computed_value)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

C_SOUND = 343.0  # m/s — speed of sound at 20 °C
_THROAT_CHAMBER_VOLUME_L = 0.1  # litres — default estimated throat chamber volume


# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class HornSegmentResult:
    """Result of Horn Segment Wizard computation."""

    #: Name of the computed parameter: "s1_cm2" | "s2_cm2" | "l12_cm" | "f12_hz"
    computed_param: str
    #: Computed value (Hz for f12, cm² for areas, cm for length)
    computed_value: float
    #: Area profile: list of (position_fraction_0_to_1, area_cm2)
    area_profile: list[tuple[float, float]]
    #: Estimated system volume in litres (horn internal + throat chamber)
    system_volume_l: float


# ─── Core geometry helpers ────────────────────────────────────────────────────

def _catenoidal_area_at(s1_m2: float, s2_m2: float, l_m: float, x_m: float) -> float:
    """Catenoidal (T=1) horn area at distance x from throat.

    A(x) = S1 × cosh²(m·x)  where  m·L = arccosh(√(S2/S1))

    Correct catenoidal (cosh²) profile: expands monotonically from S1 at the
    throat (x=0) to S2 at the mouth (x=L).  Using cosh² avoids the 1/x
    singularity near x=0 that coth² would produce.

    Boundary check:
      x=0  → cosh(0)=1 → A(0)  = S1·1   = S1  ✓
      x=L  → cosh(u_total) = √(S2/S1) → A(L) = S1·(S2/S1) = S2  ✓
    """
    if abs(s1_m2 - s2_m2) < 1e-12:
        return s1_m2
    if x_m <= 0.0:
        return s1_m2
    area_ratio_sqrt = math.sqrt(s2_m2 / s1_m2)
    u_total = math.acosh(area_ratio_sqrt)
    m = u_total / l_m
    cosh_val = math.cosh(m * x_m)
    return s1_m2 * cosh_val * cosh_val


def _catenoidal_horn_volume_l(s1_m2: float, s2_m2: float, l_m: float) -> float:
    """Catenoidal horn internal volume in litres.

    For an expanding horn (S2 > S1):
      V = S1/(2m) · [sinh(u_total)·cosh(u_total) − u_total]
      where  u_total = arccosh(√(S2/S1)) > 0,  m = u_total/L

    For a cylindrical / contracting horn (S2 ≤ S1):
      V ≈ S1 × L  (simple cylindrical approximation)
    """
    if s2_m2 <= s1_m2:
        return max(s1_m2 * l_m * 1000.0, 0.0)
    area_ratio_sqrt = math.sqrt(s2_m2 / s1_m2)
    if area_ratio_sqrt <= 1.0 + 1e-12:
        return max(s1_m2 * l_m * 1000.0, 0.0)
    u_total = math.acosh(area_ratio_sqrt)
    m = u_total / l_m
    v_m3 = (s1_m2 / (2.0 * m)) * (math.sinh(u_total) * math.cosh(u_total) - u_total)
    return max(v_m3, 0.0) * 1000.0


# ─── Main compute function ─────────────────────────────────────────────────────

def compute_horn_segment(
    s1_m2: Optional[float] = None,
    s2_m2: Optional[float] = None,
    l12_m: Optional[float] = None,
    f12_hz: Optional[float] = None,
) -> HornSegmentResult:
    """Compute the missing 4th parameter for a catenoidal horn segment.

    Exactly 3 of the 4 parameters must be provided (not None).
    Areas are in m², length in m, frequency in Hz.

    Parameters
    ----------
    s1_m2 : float | None
        Throat (neck) area in m².
    s2_m2 : float | None
        Mouth area in m².
    l12_m : float | None
        Horn axis length in m.
    f12_hz : float | None
        Low-frequency cutoff (-3 dB) in Hz.

    Returns
    -------
    HornSegmentResult
        Contains the computed parameter name and value, 20-point area profile,
        and estimated system volume.

    Raises
    ------
    ValueError
        If not exactly 3 parameters are provided, or if values are invalid.
    """
    s1 = s1_m2
    s2 = s2_m2
    l12 = l12_m
    f12 = f12_hz

    provided = sum(1 for v in (s1, s2, l12, f12) if v is not None)
    if provided != 3:
        raise ValueError(
            f"Exactly 3 of (s1_m2, s2_m2, l12_m, f12_hz) must be provided; got {provided}"
        )

    # ── Case dispatch ─────────────────────────────────────────────────────────
    if s1 is not None and s2 is not None and l12 is not None:
        # → compute F12
        if s1 <= 0 or s2 <= 0 or l12 <= 0:
            raise ValueError("s1, s2, l12 must be positive")
        if s1 >= s2:
            raise ValueError("s1 must be < s2 for an expanding horn")
        ratio = s2 / s1
        computed_value = C_SOUND / (2.0 * math.pi * l12) * math.sqrt(ratio - 1.0)
        computed_param = "f12_hz"
        s1_u, s2_u, l_u = s1, s2, l12

    elif s1 is not None and s2 is not None and f12 is not None:
        # → compute L12
        if s1 <= 0 or s2 <= 0 or f12 <= 0:
            raise ValueError("s1, s2, f12 must be positive")
        if s1 >= s2:
            raise ValueError("s1 must be < s2 for an expanding horn")
        ratio = s2 / s1
        l_u = C_SOUND / (2.0 * math.pi * f12) * math.sqrt(ratio - 1.0)
        computed_value = round(l_u * 100.0, 4)  # cm
        computed_param = "l12_cm"
        s1_u, s2_u = s1, s2

    elif s1 is not None and l12 is not None and f12 is not None:
        # → compute S2
        if s1 <= 0 or l12 <= 0 or f12 <= 0:
            raise ValueError("s1, l12, f12 must be positive")
        term = (2.0 * math.pi * f12 * l12 / C_SOUND) ** 2
        s2_u = s1 / (1.0 + term)
        if s2_u <= s1:
            raise ValueError(
                f"Computed S2 ({s2_u*1e4:.2f} cm²) ≤ S1 ({s1*1e4:.2f} cm²) — "
                "this combination produces a contracting (or cylindrical) horn, "
                "not an expanding horn. For an expanding catenoidal horn S2 must be > S1. "
                "Try a lower cutoff frequency, a shorter horn, or compute a different parameter."
            )
        computed_value = round(s2_u * 1e4, 4)  # cm²
        computed_param = "s2_cm2"
        s1_u, l_u = s1, l12

    elif s2 is not None and l12 is not None and f12 is not None:
        # → compute S1
        if s2 <= 0 or l12 <= 0 or f12 <= 0:
            raise ValueError("s2, l12, f12 must be positive")
        term = (2.0 * math.pi * f12 * l12 / C_SOUND) ** 2
        s1_u = s2 / (1.0 + term)
        computed_value = round(s1_u * 1e4, 4)  # cm²
        computed_param = "s1_cm2"
        s2_u, l_u = s2, l12

    else:
        raise ValueError("Invalid parameter combination")

    # ── Area profile (20 points including throat and mouth) ───────────────────
    N = 20
    area_profile: list[tuple[float, float]] = []
    for i in range(N + 1):
        frac = i / N
        x_m = frac * l_u
        area_m2 = _catenoidal_area_at(s1_u, s2_u, l_u, x_m)
        area_profile.append((round(frac, 4), round(area_m2 * 1e4, 4)))

    # ── System volume estimate ─────────────────────────────────────────────────
    horn_vol_l = _catenoidal_horn_volume_l(s1_u, s2_u, l_u)
    system_volume_l = round(horn_vol_l + _THROAT_CHAMBER_VOLUME_L, 4)

    return HornSegmentResult(
        computed_param=computed_param,
        computed_value=computed_value,
        area_profile=area_profile,
        system_volume_l=system_volume_l,
    )
