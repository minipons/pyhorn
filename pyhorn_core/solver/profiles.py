"""
Horn cross-sectional profile geometry.

Provides analytic formulas and discretisation routines for the standard horn
profile families used in loudspeaker design:

================  ===========================================================
Profile type     Key reference
================  ===========================================================
straight / trx   Cylindrical pipe — no flare, no cutoff
conical / con    Linear radius expansion (keeps k·r constant at cutoff)
exponential / exp  Exponential area expansion — classic Tractrix approx.
parabolic / par  Linear area expansion
hyperbolic / hyp  Hyperbolic cosh² expansion; T=1 → catenoidal
catenoidal       Hyperbolic with T=1 — minimal-surface horn shape
tractrix / trx   Tangent-right horn contour (Benade 1988 approx.)
================  ===========================================================

Profile area at any axial position is given by
``A(x) = throat_area × f(x/length)`` where *f* depends on the profile type.

Public API
----------
profile_area_at_distance()   — analytic A(x) at a single point
horn_profile_metrics()       — cutoff frequency, k·r parameters, mouth rating
discretise_profile()         — split a continuous profile into N cylindrical
                               segments suitable for TMM transmission-line
                               cascade
discretise_conical_segments()  — convert ``[dim_start, dim_end, length]``
                                 per-segment tables to TMM segment list
discretise_rectangular_segments() — same for 2-D rectangular cross-sections
"""
import math

import numpy as np
from typing import List, Tuple, Optional, Dict

# Physical constants
_RHO = 1.21  # Air density kg/m³
_C = 343.0  # Speed of sound m/s


def profile_area_at_distance(
    profile_type: str,
    throat_area: float,
    mouth_area: float,
    length: float,
    distance: float,
    hyperbolic_t: float = 1.0,
) -> float:
    """Return horn cross-sectional area at a distance along the path."""
    if length <= 0:
        raise ValueError("Length must be > 0")
    if throat_area <= 0 or mouth_area <= 0:
        raise ValueError("Areas must be > 0")

    x = float(np.clip(distance, 0.0, length))
    profile_key = profile_type.lower()

    if profile_key in ("con", "conical"):
        rt = math.sqrt(throat_area / math.pi)
        rm = math.sqrt(mouth_area / math.pi)
        radius = rt + (rm - rt) * (x / length)
        return float(math.pi * radius**2)

    if profile_key in ("exp", "exponential"):
        m = (1.0 / length) * math.log(mouth_area / throat_area)
        return float(throat_area * math.exp(m * x))

    if profile_key in ("par", "parabolic"):
        return float(throat_area + (mouth_area - throat_area) * (x / length))

    if profile_key in ("hyp", "hyperbolic"):
        area_ratio_sqrt = math.sqrt(mouth_area / throat_area)
        u_total = _solve_hyperbolic_u(area_ratio_sqrt, hyperbolic_t)
        m = u_total / length
        return float(
            throat_area * (math.cosh(m * x) + hyperbolic_t * math.sinh(m * x)) ** 2
        )

    raise ValueError(f"Unknown profile type: {profile_type}")


def horn_profile_metrics(
    profile_type: str,
    throat_area: float,
    mouth_area: float,
    length: float,
    hyperbolic_t: float = 1.0,
) -> Dict[str, float | str]:
    """
    Compute horn-acoustic metrics for a continuous expansion profile.

    Returns a dict with:
      flare_constant_m   — m  (1/m)
      cutoff_hz          — fc (Hz); 0 for conical (no cutoff in this sense)
      krm                — k·rm at cutoff (dimensionless)
      mouth_radius_m     — rm = sqrt(mouth_area/π)
      mouth_diameter_cm  — 2·rm in cm
      expansion_ratio    — mouth_area / throat_area
      mouth_rating       — 'midrange_ok' | 'bass_ok' | 'undersized'
      mouth_krm_min_hz   — freq below which krm < 0.7 (Keele's bass threshold)

    Rating criteria (krm = k·rm, k = 2πf/c):
      krm ≥ 1.0  → 'midrange_ok'   (minimal ripple per Keele 1973)
      krm ≥ 0.7  → 'bass_ok'       (smooth response, Keele optimum)
      krm < 0.7  → 'undersized'    (avoid for bass; mouth reflections significant)

    Refs:
      - Keele, D.B. (1973). "Optimum Horn Mouth Size." AES Preprint 933.
      - Kolbrek, B. (2008). "Horn Theory: An Introduction." audioXpress.
    """
    if throat_area <= 0 or mouth_area <= 0 or length <= 0:
        return {
            k: 0.0
            for k in (
                "flare_constant_m",
                "cutoff_hz",
                "krm",
                "kaL",
                "mouth_radius_m",
                "mouth_diameter_cm",
                "expansion_ratio",
                "mouth_rating",
                "mouth_krm_min_hz",
                "tl_tuning_hz",
                "mouth_ko",
            )
        }

    pt = profile_type.lower()
    rm = np.sqrt(mouth_area / np.pi)

    if pt in ("con", "conical"):
        # Conical horns: no exponential cutoff per se.
        # Treat flare constant as 0; cutoff_hz=0.
        # For krm diagnostics, approximate as m=0 (flat).
        m = 0.0
    elif pt in ("exp", "exponential"):
        m = (1.0 / length) * np.log(mouth_area / throat_area)
    elif pt in ("par", "parabolic"):
        # Linear area expansion — not a true exponential horn.
        ratio = mouth_area / throat_area
        m = (1.0 / length) * np.log(ratio)
    elif pt in ("hyp", "hyperbolic"):
        area_ratio_sqrt = float(np.sqrt(mouth_area / throat_area))
        m = _solve_hyperbolic_u(area_ratio_sqrt, hyperbolic_t) / length
    elif pt in ("tractrix", "trx"):
        # Tractrix: uses constant-radius spherical wave-fronts.
        # Approximate effective flare constant from area ratio.
        m = (1.0 / length) * np.log(mouth_area / throat_area)
    elif pt in ("straight",):
        # Straight / cylindrical: constant area, no flare, no exponential cutoff.
        m = 0.0
    else:
        m = 0.0

    # Cutoff frequency: fc = (m·c) / (4π) — standard exponential cutoff.
    fc = (m * _C) / (4.0 * np.pi) if m > 0 else 0.0
    if pt in ("hyp", "hyperbolic"):
        fc = (m * _C) / (2.0 * np.pi) if m > 0 else 0.0

    # Mouth wave-number parameter at fc: krm_fc = (2πfc/c)·rm = m·rm/2
    krm_fc = (rm * m) / 2.0 if m > 0 else float("inf")
    if pt in ("hyp", "hyperbolic"):
        krm_fc = rm * m if m > 0 else float("inf")

    # Frequency where krm = 0.7 (Keele's lower limit for bass horns)
    mouth_krm_min_hz = (0.7 * _C) / (2.0 * np.pi * rm) if rm > 0 else 0.0

    if krm_fc >= 1.0:
        mouth_rating = "midrange_ok"
    elif krm_fc >= 0.7:
        mouth_rating = "bass_ok"
    else:
        mouth_rating = "undersized"

    return {
        "flare_constant_m": float(m),
        "cutoff_hz": float(fc),
        "krm": float(krm_fc),  # dimensionless mouth parameter at cutoff: k·rm = m·rm/2
        "mouth_radius_m": float(rm),
        "mouth_diameter_cm": float(2.0 * rm * 100.0),
        "expansion_ratio": float(mouth_area / throat_area),
        "mouth_rating": mouth_rating,
        "mouth_krm_min_hz": float(mouth_krm_min_hz),
        "tl_tuning_hz": float(_C / (4.0 * length)) if length > 0 else 0.0,
        "mouth_ko": float(2.0 * rm),  # mouth dimension (square side or circle diameter)
    }


def discretise_profile(
    profile_type: str,
    throat_area: float,
    mouth_area: float,
    length: float,
    n_segments: int = 100,
    hyperbolic_t: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Generate an array of discrete cylindrical segments for a given continuous expansion profile.
    Returns: List of (segment_length_m, segment_area_m2)
    """
    if length <= 0:
        raise ValueError("Length must be > 0")
    if throat_area <= 0 or mouth_area <= 0:
        raise ValueError("Areas must be > 0")

    profile_type = profile_type.lower()
    dx = length / n_segments
    x = np.linspace(0, length, n_segments + 1)

    if profile_type in ("con", "conical"):
        # Radius expands linearly
        rt = np.sqrt(throat_area / np.pi)
        rm = np.sqrt(mouth_area / np.pi)
        r_x = rt + (rm - rt) * (x / length)
        A_x = np.pi * r_x**2

    elif profile_type in ("exp", "exponential"):
        # Area expands exponentially
        # A(x) = At * e^(mx)
        m = (1.0 / length) * np.log(mouth_area / throat_area)
        A_x = throat_area * np.exp(m * x)

    elif profile_type in ("par", "parabolic"):
        # Linear area expansion (radius grows as sqrt, hence "parabolic" profile)
        A_x = throat_area + (mouth_area - throat_area) * (x / length)

    elif profile_type in ("hyp", "hyperbolic", "catenoidal"):
        # catenoidal is hyperbolic with T=1 (the natural catenoid shape = minimal surface for given boundary)
        area_ratio_sqrt = float(np.sqrt(mouth_area / throat_area))
        u_total = _solve_hyperbolic_u(area_ratio_sqrt, hyperbolic_t)
        m = u_total / length
        A_x = throat_area * (np.cosh(m * x) + hyperbolic_t * np.sinh(m * x)) ** 2

    elif profile_type in ("tractrix", "trx"):
        # Tractrix horn: the horn contour follows a tractrix curve such that
        # a line tangent to the wall is always perpendicular to the wave-front.
        #
        # Parametric form (Benade 1988, Kolbrek 2008):
        #   r_w(θ) = a / cosh(θ)      [tractrix wall radius]
        #   x(θ)   = a · (θ - tanh θ) [axial distance]
        # where a = throat radius = sqrt(throat_area/π).
        #
        # For an expanding horn (rm > rt), the mouth condition
        # r_w(θ_m) = rm·sin(θ_m) must be solved numerically — the
        # "folded tractrix" has the axis following a curved path.
        #
        # Simplified implementation: tanh-based smooth expansion from rt to rm.
        # Captures the key property: rapid initial flare, rate slowing toward mouth.
        # k_tr=0.3 gives smooth tractrix-like curvature (gentler than exponential).
        rt = np.sqrt(throat_area / np.pi)
        rm = np.sqrt(mouth_area / np.pi)
        k_tr = 0.3
        t = np.tanh(k_tr * x / length)
        r_x = rt + (rm - rt) * t / np.tanh(k_tr)
        r_x = np.clip(r_x, rt, rm)
        A_x = np.pi * r_x**2

    elif profile_type in ("straight",):
        # Constant-area cylinder / straight throat: no flare.
        # Physically this is a short tube — no low-frequency cutoff from the straight section.
        A_x = np.full_like(x, throat_area, dtype=float)

    else:
        raise ValueError(
            f"Unknown profile type: {profile_type}. "
            f"Supported: straight, conical, exponential, parabolic, hyperbolic, tractrix, catenoidal"
        )

    # Create segments using average area of each slice
    segments = []
    for i in range(n_segments):
        A_avg = (A_x[i] + A_x[i + 1]) / 2.0
        segments.append((dx, float(A_avg), 0.0))

    return segments


def _solve_hyperbolic_u(area_ratio_sqrt: float, hyperbolic_t: float) -> float:
    if area_ratio_sqrt <= 0:
        raise ValueError("sqrt(mouth_area/throat_area) must be > 0")

    if abs(hyperbolic_t + 1.0) <= 1e-12:
        if area_ratio_sqrt >= 1.0:
            raise ValueError("hyperbolic_t = -1 requires mouth_area < throat_area")
        return float(-math.log(area_ratio_sqrt))

    discriminant = area_ratio_sqrt**2 + hyperbolic_t**2 - 1.0
    if discriminant < -1e-12:
        raise ValueError("Invalid hyperbolic profile parameters")

    root = math.sqrt(max(discriminant, 0.0))
    denom = 1.0 + hyperbolic_t
    candidates = []
    for numerator in (area_ratio_sqrt + root, area_ratio_sqrt - root):
        y = numerator / denom
        if y > 0:
            candidates.append(y)

    if not candidates:
        raise ValueError("Invalid hyperbolic profile parameters")

    for candidate in sorted(candidates):
        if candidate > 1.0 + 1e-12:
            return float(math.log(candidate))

    raise ValueError("Invalid hyperbolic profile parameters")


def discretise_conical_segments(
    conical_segments: List[Tuple[float, ...]],
    width: Optional[float] = None,
    n_per_segment: int = 10,
) -> Tuple[List[Tuple[float, ...]], List[Tuple[float, float]]]:
    """
    Takes a list of [dim_start, dim_end, length].
    If width is set, dims are treated as heights and converted to area.
    Returns: (segments, bends) where:
    - segments is a list of (length, average_area) for transmission lines
    - bends is a list of (area_before, area_after) for impedance mismatches at the joints
    """
    segments = []
    bends = []

    last_area_end = None

    for seg in conical_segments:
        if len(seg) < 3:
            raise ValueError(
                f"Each conical segment must have at least 3 elements [dim_start, dim_end, length], got {seg}"
            )
        if len(seg) >= 4:
            dim_start, dim_end, length, fr = seg[0], seg[1], seg[2], seg[3]
        else:
            dim_start, dim_end, length = seg[0], seg[1], seg[2]
            fr = 0.0

        if width is not None:
            area_start = dim_start * width
            area_end = dim_end * width
        else:
            area_start = dim_start
            area_end = dim_end

        # Check for bend/elbow from the last segment
        if last_area_end is not None and abs(last_area_end - area_start) > 1e-6:
            bends.append((last_area_end, area_start))

        # Discretise this section using linear area expansion
        # For folded horns with constant width, Area = height * width, and height expands linearly.
        # So Area expands linearly (equivalent to parabolic profile).
        dx = length / n_per_segment
        x = np.linspace(0, length, n_per_segment + 1)
        A_x = area_start + (area_end - area_start) * (x / length)

        for i in range(n_per_segment):
            A_avg = (A_x[i] + A_x[i + 1]) / 2.0
            segments.append((dx, float(A_avg), float(fr)))

        last_area_end = area_end

    return segments, bends


def discretise_rectangular_segments(
    rectangular_segments: List[Tuple[float, ...]],
    n_per_segment: int = 10,
) -> Tuple[List[Tuple[float, ...]], List[Tuple[float, float]], List[float]]:
    """
    Takes a list of [width_start, height_start, width_end, height_end, length, [fr]].

    Returns:
    - segments: list of (length, average_area, fr)
    - bends: list of (area_before, area_after) at segment joints
    - node_widths: width at each discretised segment node for 3D plotting
    """
    segments = []
    bends = []
    node_widths: List[float] = []

    last_area_end = None

    for seg_idx, seg in enumerate(rectangular_segments):
        if len(seg) < 5:
            raise ValueError(
                "Each rectangular segment must have at least 5 elements "
                "[width_start, height_start, width_end, height_end, length]"
            )

        if len(seg) >= 6:
            width_start, height_start, width_end, height_end, length, fr = seg[:6]
        else:
            width_start, height_start, width_end, height_end, length = seg[:5]
            fr = 0.0

        area_start = width_start * height_start
        area_end = width_end * height_end

        if last_area_end is not None and abs(last_area_end - area_start) > 1e-6:
            bends.append((last_area_end, area_start))

        dx = length / n_per_segment
        x = np.linspace(0, length, n_per_segment + 1)
        widths = width_start + (width_end - width_start) * (x / length)
        heights = height_start + (height_end - height_start) * (x / length)
        areas = widths * heights

        if seg_idx == 0:
            node_widths.append(float(widths[0]))
        for i in range(n_per_segment):
            A_avg = (areas[i] + areas[i + 1]) / 2.0
            segments.append((dx, float(A_avg), float(fr)))
            node_widths.append(float(widths[i + 1]))

        last_area_end = area_end

    return segments, bends, node_widths
