"""Solve Hornresp profile parameters for all supported profile types.

Supports: Con (conical), Exp (exponential), Par (parabolic),
Cat (catenoidal), Hyp (hyperbolic).

Profile types and their Hornresp parameterizations:

  Con (conical):
    Area(x) = S1 + (S2−S1)·(x/L12)   [linear radius expansion]
    No exponential cutoff — pure geometric expansion.
    Given S1, S2, L12 → direct solve (F12 is N/A for conical).

  Exp (exponential):
    Area(x) = S1 · exp(m·x),  m = (1/L12)·ln(S2/S1)
    F12 = c·m/(4π)  (cutoff frequency)
    Given S1, S2, F12 → L12 = c/(4π·F12)·ln(S2/S1)
    Given S1, S2, L12 → F12 = c/(4π·L12)·ln(S2/S1)

  Par (parabolic):
    Area(x) = S1·(1−(1−√(S2/S1))·x/L12)²  [quadratic expansion]
    No exponential cutoff in the Hornresp sense.
    Given S1, S2, L12 → direct solve (F12 is N/A for parabolic).

  Cat (catenoidal):
    Same as hyperbolic with T=1: Area(x) = S1·cosh²(m·x),  m·L12 = arccosh(√(S2/S1))
    Given S1, S2, L12 → direct (hyperbolic solver with T=1).
    F12 derived from catenoidal m.

  Hyp (hyperbolic):
    Area(x) = S1·(cosh(m·x) + T·sinh(m·x))²
    Given S1, S2, F12, T, Hyp → direct solve.
    Any one parameter may be omitted and solved from the other four.

References:
  - Hornresp manual pages 044-046 (Normal Horn — Con/Exp/Par profiles)
  - Kolbrek, B. (2008). "Horn Theory: An Introduction." audioXpress.
"""

import math
from typing import Dict, Optional, Union


_C = 343.0
_EPS = 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _solve_u_from_ratio_and_t(area_ratio_sqrt: float, t: float) -> float:
    """Solve u = m·L from area ratio sqrt(S2/S1) and hyperbolic T parameter."""
    if area_ratio_sqrt <= 0:
        raise ValueError("sqrt(S2/S1) must be > 0")

    if abs(t + 1.0) <= _EPS:
        if area_ratio_sqrt >= 1.0:
            raise ValueError("T = -1 is only valid when S2 < S1")
        return -math.log(area_ratio_sqrt)

    discriminant = area_ratio_sqrt**2 + t**2 - 1.0
    if discriminant < -_EPS:
        raise ValueError(
            "Hornresp inputs are inconsistent for a real hyperbolic solution"
        )

    root = math.sqrt(max(discriminant, 0.0))
    denom = 1.0 + t
    candidates = []
    for numerator in (area_ratio_sqrt + root, area_ratio_sqrt - root):
        y = numerator / denom
        if y > 0:
            candidates.append(y)

    if not candidates:
        raise ValueError("Hornresp inputs do not yield a positive hyperbolic solution")

    for candidate in sorted(candidates):
        if candidate > 1.0 + _EPS:
            return math.log(candidate)

    best = max(candidates)
    if best <= 1.0 + _EPS:
        raise ValueError(
            "Hornresp inputs imply a non-positive horn rate; "
            "check S1, S2, T, F12, and Hyp"
        )
    return math.log(best)


def _hornresp_hyperbolic_ratio(f12_hz: float, hyp_cm: float, t: float) -> float:
    """Return sqrt(S2/S1) for a hyperbolic horn given F12, Hyp, T."""
    _validate_positive("F12", f12_hz)
    _validate_positive("Hyp", hyp_cm)
    length_m = hyp_cm / 100.0
    u = (2.0 * math.pi * f12_hz * length_m) / _C
    return math.cosh(u) + t * math.sinh(u)


# ─────────────────────────────────────────────────────────────────────────────
# Per-profile solvers
# ─────────────────────────────────────────────────────────────────────────────


def _solve_conical(
    s1_cm2: Optional[float] = None,
    s2_cm2: Optional[float] = None,
    l12_cm: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """Solve conical horn: Area(x) = S1 + (S2−S1)·(x/L12).

    Conical horns have no exponential cutoff frequency — purely geometric.
    Requires: S1, S2, L12 (all three).
    F12 is set to 0 (N/A for conical profiles).
    """
    missing = [n for n, v in [("s1", s1_cm2), ("s2", s2_cm2), ("l12", l12_cm)]
               if v is None]
    if missing:
        raise ValueError(
            f"Conical profile requires S1, S2, and L12. Missing: {', '.join(missing)}."
        )

    _validate_positive("S1", s1_cm2)
    _validate_positive("S2", s2_cm2)
    _validate_positive("L12", l12_cm)

    l12_m = l12_cm / 100.0
    # Conical: no F12 (set to 0 — N/A for pure cone)
    return {
        "s1_cm2": float(s1_cm2),
        "s2_cm2": float(s2_cm2),
        "l12_cm": float(l12_cm),
        "f12_hz": 0.0,
        "t": 0.0,
        "hyp_cm": float(l12_cm),
        "path_length_m": float(l12_m),
        "throat_area_m2": float(s1_cm2 / 10000.0),
        "mouth_area_m2": float(s2_cm2 / 10000.0),
        "u": 0.0,
        "flare_constant_m": 0.0,
        "profile_type": "conical",
    }


def _solve_exponential(
    s1_cm2: Optional[float] = None,
    s2_cm2: Optional[float] = None,
    f12_hz: Optional[float] = None,
    l12_cm: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """Solve exponential horn: Area(x) = S1·exp(m·x),  m = (1/L12)·ln(S2/S1).

    F12 = c·m/(4π) is the low-frequency cutoff.
    Two of {F12, L12} may be provided; the other is solved.

    Given S1, S2, F12 → L12 = c/(4π·F12)·ln(S2/S1)
    Given S1, S2, L12 → F12 = c/(4π·L12)·ln(S2/S1)
    Given F12, L12 → verify S1/S2 consistency
    """
    provided = {
        "s1": s1_cm2,
        "s2": s2_cm2,
        "f12": f12_hz,
        "l12": l12_cm,
    }
    missing = [n for n, v in provided.items() if v is None]

    if len(missing) >= 2:
        raise ValueError(
            f"Exponential profile requires at least two of S1, S2, F12, L12. "
            f"Missing: {', '.join(missing)}."
        )

    for key, val in [("S1", s1_cm2), ("S2", s2_cm2)]:
        if val is not None:
            _validate_positive(key, val)

    area_ratio = s2_cm2 / s1_cm2 if s1_cm2 and s2_cm2 else None
    ln_ratio = math.log(area_ratio) if area_ratio and area_ratio > 0 else None

    if f12_hz is not None and l12_cm is not None:
        # Both F12 and L12 provided — verify or compute S1/S2
        _validate_positive("F12", f12_hz)
        _validate_positive("L12", l12_cm)
        l12_m = l12_cm / 100.0
        f12_hz = float(f12_hz)
        l12_cm = float(l12_cm)
        if s1_cm2 is not None and s2_cm2 is not None:
            expected_f12 = _C * ln_ratio / (4.0 * math.pi * l12_m) if ln_ratio else 0.0
            # Allow 1% tolerance
            if not math.isclose(expected_f12, f12_hz, rel_tol=0.01):
                raise ValueError(
                    f"Provided F12={f12_hz:.2f} Hz is inconsistent with "
                    f"S1={s1_cm2}, S2={s2_cm2}, L12={l12_cm} cm. "
                    f"Expected F12≈{expected_f12:.2f} Hz."
                )
        else:
            # Compute one of S1, S2 from the other + F12 + L12
            if s1_cm2 is not None and s2_cm2 is None:
                # L12 = c/(4π·F12)·ln(S2/S1)  →  S2 = S1·exp(4π·F12·L12/c)
                s2_cm2 = s1_cm2 * math.exp(4.0 * math.pi * f12_hz * l12_m / _C)
            elif s2_cm2 is not None and s1_cm2 is None:
                s1_cm2 = s2_cm2 / math.exp(4.0 * math.pi * f12_hz * l12_m / _C)
            else:
                raise ValueError(
                    "Provide at least S1 or S2 alongside F12 and L12."
                )

    elif f12_hz is not None:
        # F12 given, L12 missing → compute L12
        _validate_positive("F12", f12_hz)
        _validate_positive("S1", s1_cm2)
        _validate_positive("S2", s2_cm2)
        l12_m = _C * ln_ratio / (4.0 * math.pi * f12_hz)
        l12_cm = l12_m * 100.0

    elif l12_cm is not None:
        # L12 given, F12 missing → compute F12
        _validate_positive("L12", l12_cm)
        _validate_positive("S1", s1_cm2)
        _validate_positive("S2", s2_cm2)
        l12_m = l12_cm / 100.0
        f12_hz = _C * ln_ratio / (4.0 * math.pi * l12_m)

    else:
        # Both F12 and L12 missing — use S1, S2 (underdetermined without one of F12/L12)
        raise ValueError(
            "Provide at least two of S1, S2, F12, L12 for exponential profile."
        )

    l12_m = l12_cm / 100.0
    m = ln_ratio / l12_m  # flare constant (1/m)
    u = m * l12_m  # = ln_ratio

    return {
        "s1_cm2": float(s1_cm2),
        "s2_cm2": float(s2_cm2),
        "l12_cm": float(l12_cm),
        "f12_hz": float(f12_hz),
        "t": 0.0,
        "hyp_cm": float(l12_cm),
        "path_length_m": float(l12_m),
        "throat_area_m2": float(s1_cm2 / 10000.0),
        "mouth_area_m2": float(s2_cm2 / 10000.0),
        "u": float(u),
        "flare_constant_m": float(m),
        "profile_type": "exponential",
    }


def _solve_parabolic(
    s1_cm2: Optional[float] = None,
    s2_cm2: Optional[float] = None,
    l12_cm: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """Solve parabolic horn: Area(x) = S1·(1−(1−√(S2/S1))·x/L12)².

    Parabolic horns have no exponential cutoff — purely geometric.
    Requires: S1, S2, L12 (all three).
    F12 is set to 0 (N/A for parabolic profiles).
    """
    missing = [n for n, v in [("s1", s1_cm2), ("s2", s2_cm2), ("l12", l12_cm)]
               if v is None]
    if missing:
        raise ValueError(
            f"Parabolic profile requires S1, S2, and L12. Missing: {', '.join(missing)}."
        )

    _validate_positive("S1", s1_cm2)
    _validate_positive("S2", s2_cm2)
    _validate_positive("L12", l12_cm)

    l12_m = l12_cm / 100.0
    return {
        "s1_cm2": float(s1_cm2),
        "s2_cm2": float(s2_cm2),
        "l12_cm": float(l12_cm),
        "f12_hz": 0.0,
        "t": 0.0,
        "hyp_cm": float(l12_cm),
        "path_length_m": float(l12_m),
        "throat_area_m2": float(s1_cm2 / 10000.0),
        "mouth_area_m2": float(s2_cm2 / 10000.0),
        "u": 0.0,
        "flare_constant_m": 0.0,
        "profile_type": "parabolic",
    }


def _solve_catenoidal(
    s1_cm2: Optional[float] = None,
    s2_cm2: Optional[float] = None,
    l12_cm: Optional[float] = None,
    f12_hz: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """Solve catenoidal horn: Area(x) = S1·cosh²(m·x),  m·L12 = arccosh(√(S2/S1)).

    Catenoidal = hyperbolic with T=1. This is the straightest catenary curve
    between two circles — the optimal shape for minimum-length horns.
    F12 is derived from m: F12 = c·m/(2π).
    """
    missing = [n for n, v in [("s1", s1_cm2), ("s2", s2_cm2), ("l12", l12_cm)]
               if v is None]
    if missing:
        raise ValueError(
            f"Catenoidal profile requires S1, S2, and L12. Missing: {', '.join(missing)}."
        )

    _validate_positive("S1", s1_cm2)
    _validate_positive("S2", s2_cm2)
    _validate_positive("L12", l12_cm)

    l12_m = l12_cm / 100.0
    area_ratio_sqrt = math.sqrt(s2_cm2 / s1_cm2)
    if area_ratio_sqrt <= 1.0:
        raise ValueError("S2 must be > S1 for a catenoidal expansion horn.")
    u_total = math.acosh(area_ratio_sqrt)
    m = u_total / l12_m
    f12_hz = _C * m / (2.0 * math.pi)

    return {
        "s1_cm2": float(s1_cm2),
        "s2_cm2": float(s2_cm2),
        "l12_cm": float(l12_cm),
        "f12_hz": float(f12_hz),
        "t": 1.0,
        "hyp_cm": float(l12_cm),
        "path_length_m": float(l12_m),
        "throat_area_m2": float(s1_cm2 / 10000.0),
        "mouth_area_m2": float(s2_cm2 / 10000.0),
        "u": float(u_total),
        "flare_constant_m": float(m),
        "profile_type": "catenoidal",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Hyperbolic (original implementation, kept for direct calls)
# ─────────────────────────────────────────────────────────────────────────────


def solve_hornresp_hyperbolic(
    s1_cm2: Optional[float] = None,
    s2_cm2: Optional[float] = None,
    f12_hz: Optional[float] = None,
    t: Optional[float] = None,
    hyp_cm: Optional[float] = None,
) -> Dict[str, float]:
    """Solve Hornresp hyperbolic-exponential horn parameters.

    Exactly one of S1, S2, F12, T, or Hyp may be omitted and will be solved
    from the other four. Two omitted inputs are underdetermined and rejected.
    """
    provided = {
        "s1": s1_cm2,
        "s2": s2_cm2,
        "f12": f12_hz,
        "t": t,
        "hyp": hyp_cm,
    }
    missing = [name for name, value in provided.items() if value is None]

    if len(missing) >= 2:
        raise ValueError(
            "Two missing Hornresp inputs are underdetermined; "
            "provide at least four of S1, S2, F12, T, and Hyp."
        )

    if s1_cm2 is not None:
        _validate_positive("S1", s1_cm2)
    if s2_cm2 is not None:
        _validate_positive("S2", s2_cm2)
    if f12_hz is not None:
        _validate_positive("F12", f12_hz)
    if hyp_cm is not None:
        _validate_positive("Hyp", hyp_cm)

    if not missing:
        assert s1_cm2 is not None
        assert s2_cm2 is not None
        assert f12_hz is not None
        assert t is not None
        assert hyp_cm is not None
        predicted_ratio = _hornresp_hyperbolic_ratio(f12_hz, hyp_cm, t)
        actual_ratio = math.sqrt(s2_cm2 / s1_cm2)
        if not math.isclose(predicted_ratio, actual_ratio, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError(
                "Provided Hornresp inputs are inconsistent; "
                "they do not satisfy the hyperbolic flare equation."
            )
    else:
        missing_name = missing[0]

        if missing_name == "s1":
            assert s2_cm2 is not None
            assert f12_hz is not None
            assert t is not None
            assert hyp_cm is not None
            ratio = _hornresp_hyperbolic_ratio(f12_hz, hyp_cm, t)
            s1_cm2 = s2_cm2 / (ratio**2)
        elif missing_name == "s2":
            assert s1_cm2 is not None
            assert f12_hz is not None
            assert t is not None
            assert hyp_cm is not None
            ratio = _hornresp_hyperbolic_ratio(f12_hz, hyp_cm, t)
            s2_cm2 = s1_cm2 * (ratio**2)
        elif missing_name == "t":
            assert s1_cm2 is not None
            assert s2_cm2 is not None
            assert f12_hz is not None
            assert hyp_cm is not None
            length_m = hyp_cm / 100.0
            u = (2.0 * math.pi * f12_hz * length_m) / _C
            sinh_u = math.sinh(u)
            if abs(sinh_u) <= _EPS:
                raise ValueError("Cannot solve T when F12 and Hyp imply zero horn rate")
            area_ratio_sqrt = math.sqrt(s2_cm2 / s1_cm2)
            t = (area_ratio_sqrt - math.cosh(u)) / sinh_u
        else:
            assert s1_cm2 is not None
            assert s2_cm2 is not None
            assert t is not None
            area_ratio_sqrt = math.sqrt(s2_cm2 / s1_cm2)
            u = _solve_u_from_ratio_and_t(area_ratio_sqrt, t)
            if missing_name == "f12":
                assert hyp_cm is not None
                length_m = hyp_cm / 100.0
                f12_hz = (_C * u) / (2.0 * math.pi * length_m)
            else:
                assert f12_hz is not None
                hyp_cm = 100.0 * (_C * u) / (2.0 * math.pi * f12_hz)

    assert s1_cm2 is not None
    assert s2_cm2 is not None
    assert f12_hz is not None
    assert t is not None
    assert hyp_cm is not None

    area_ratio_sqrt = math.sqrt(s2_cm2 / s1_cm2)
    u = _solve_u_from_ratio_and_t(area_ratio_sqrt, t)

    return {
        "s1_cm2": float(s1_cm2),
        "s2_cm2": float(s2_cm2),
        "f12_hz": float(f12_hz),
        "t": float(t),
        "hyp_cm": float(hyp_cm),
        "path_length_m": float(hyp_cm / 100.0),
        "throat_area_m2": float(s1_cm2 / 10000.0),
        "mouth_area_m2": float(s2_cm2 / 10000.0),
        "u": float(u),
        "flare_constant_m": float(u / (hyp_cm / 100.0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatcher
# ─────────────────────────────────────────────────────────────────────────────


_VALID_PROFILE_TYPES = {"con", "exp", "par", "cat", "hyp",
                        "conical", "exponential", "parabolic", "catenoidal", "hyperbolic"}


def _normalise_profile_type(pt: str) -> str:
    """Return canonical name: con | exp | par | cat | hyp."""
    mapping = {
        "conical": "con", "con": "con",
        "exponential": "exp", "exp": "exp",
        "parabolic": "par", "par": "par",
        "catenoidal": "cat", "cat": "cat",
        "hyperbolic": "hyp", "hyp": "hyp",
    }
    key = pt.lower().strip()
    if key not in mapping:
        raise ValueError(
            f"Unknown profile type '{pt}'. "
            f"Valid types: con, exp, par, cat, hyp "
            f"(or their full names: conical, exponential, parabolic, catenoidal, hyperbolic)."
        )
    return mapping[key]


def solve_hornresp_profile(
    profile_type: str,
    s1_cm2: Optional[float] = None,
    s2_cm2: Optional[float] = None,
    f12_hz: Optional[float] = None,
    t: Optional[float] = None,
    hyp_cm: Optional[float] = None,
    l12_cm: Optional[float] = None,
) -> Dict[str, Union[float, str]]:
    """Solve Hornresp parameters for any supported profile type.

    Parameters
    ----------
    profile_type:
        One of: con (conical), exp (exponential), par (parabolic),
        cat (catenoidal), hyp (hyperbolic).
    s1_cm2:
        Throat area in cm².
    s2_cm2:
        Mouth area in cm².
    f12_hz:
        Flare cutoff frequency in Hz (used for exp and hyp profiles).
    t:
        Hyperbolic T parameter (for hyp profile only).
    hyp_cm:
        Horn axis length in cm (for hyp profile; L12 is used for other profiles).
    l12_cm:
        Horn axis length in cm (alias for hyp_cm, used for con/exp/par/cat profiles).
        If both hyp_cm and l12_cm are provided, l12_cm takes precedence for
        non-hyperbolic profiles.

    Profile-specific parameter requirements
    ----------------------------------------
    con (conical):
        Requires S1, S2, L12. F12 is N/A (set to 0).
    exp (exponential):
        Requires any two of S1, S2, F12, L12.
        F12 = c·m/(4π) where m = ln(S2/S1)/L12.
    par (parabolic):
        Requires S1, S2, L12. F12 is N/A (set to 0).
    cat (catenoidal):
        Requires S1, S2, L12. F12 is derived: F12 = c·m/(2π).
        Equivalent to hyperbolic with T=1.
    hyp (hyperbolic):
        Requires any four of S1, S2, F12, T, Hyp.
        Exactly one may be omitted and solved from the other four.

    Returns
    -------
    dict with keys: s1_cm2, s2_cm2, l12_cm, f12_hz, t, hyp_cm,
    path_length_m, throat_area_m2, mouth_area_m2, u, flare_constant_m,
    profile_type
    """
    pt = _normalise_profile_type(profile_type)

    # Normalise: l12_cm is alias for hyp_cm in the API
    if l12_cm is not None:
        _validate_positive("L12", l12_cm)
    if hyp_cm is not None:
        _validate_positive("Hyp", hyp_cm)
    # For con/exp/par/cat, use l12_cm if provided, else hyp_cm if provided
    _l12 = l12_cm if l12_cm is not None else hyp_cm

    if pt == "con":
        return _solve_conical(s1_cm2=s1_cm2, s2_cm2=s2_cm2, l12_cm=_l12)
    elif pt == "exp":
        return _solve_exponential(
            s1_cm2=s1_cm2, s2_cm2=s2_cm2, f12_hz=f12_hz, l12_cm=_l12
        )
    elif pt == "par":
        return _solve_parabolic(s1_cm2=s1_cm2, s2_cm2=s2_cm2, l12_cm=_l12)
    elif pt == "cat":
        return _solve_catenoidal(
            s1_cm2=s1_cm2, s2_cm2=s2_cm2, l12_cm=_l12, f12_hz=f12_hz
        )
    elif pt == "hyp":
        # Delegate to existing hyperbolic solver
        result = solve_hornresp_hyperbolic(
            s1_cm2=s1_cm2, s2_cm2=s2_cm2, f12_hz=f12_hz, t=t, hyp_cm=hyp_cm
        )
        result["profile_type"] = "hyperbolic"
        result["l12_cm"] = result.pop("hyp_cm")  # keep l12_cm for uniform output
        return result
    # Should never reach here due to _normalise_profile_type validation
    raise ValueError(f"Unknown profile type: {profile_type}")


# ─────────────────────────────────────────────────────────────────────────────
# Public re-exports for backwards compatibility
# ─────────────────────────────────────────────────────────────────────────────

#: Alias for the internal helper — kept public for existing test imports.
hornresp_hyperbolic_ratio = _hornresp_hyperbolic_ratio
"""Return sqrt(S2/S1) for a hyperbolic horn given F12, Hyp, T.

Alias of ``_hornresp_hyperbolic_ratio()`` kept for backwards compatibility."""
