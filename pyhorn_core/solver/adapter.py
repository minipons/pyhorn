"""Throat adapter geometry designer.

Computes the minimum-length profiled transition duct between a throat chamber
(and its opening diameter D1) and the horn throat (diameter D2).

Reference: Hornresp manual page 87 (Throat Adapter Designer) and page 013
(throat adapter parameters: Ap1, Lpt).

Profile types
-------------
``cylindrical``
    Constant cross-section. D1 == D2. Length is unconstrained by geometry.
``conical``
    Linear area taper. The flare half-angle α is constant along the adapter,
    so tan(α) = |D2−D1| / (2·L).  Minimum length: L_min = |D2−D1| / (2·tan(α)).
``exponential``
    Exponential area taper.  Area at position x: A(x) = A0·exp(m·x) where
    m = ln(Ap1/A0)/L.  Minimum length derived from the peak flare rate near the
    small-diameter end: L_min ≈ 2·|ln(Ap1/A0)| / m, approximated by the conical
    minimum using A_mean = (A1+A2)/2 as the effective flare diameter.
``parabolic``
    Parabolic area taper.  √A(x) is linear in x.  Minimum length approximated
    using the mean diameter as the effective conical diameter.

Usage
-----
>>> from pyhorn_core.solver.adapter import compute_throat_adapter, throat_adapter_profile
>>> adapter = compute_throat_adapter(
...     D1=0.05, D2=0.10, A1_deg=30.0, A2_deg=30.0, profile_type="conical"
... )
>>> print(f"Minimum length: {adapter.lpt*100:.2f} cm")
>>> areas = throat_adapter_profile(adapter, n_points=101)  # area at each x
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pyhorn_core.config.chamber_models import ThroatAdapter

# ──────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class ThroatAdapterInput:
    """Input parameters for throat adapter design.

    Parameters
    ----------
    D1 :
        Input (throat-chamber side) diameter in **metres**.  This is the
        diameter of the throat chamber opening — the driver side of the adapter.
    D2 :
        Output (horn throat side) diameter in **metres**.  This is the cross-
        section presented to the first horn segment.
    A1_deg :
        Input side flare half-angle in **degrees**.  The flare angle is measured
        from the adapter axis to the conical surface.  Must be > 0 for conical,
        exponential, and parabolic profiles.
    A2_deg :
        Output side flare half-angle in **degrees**.  For conical adapters this
        should equal A1_deg.  The minimum length uses the *smaller* of A1 and A2.
    profile_type :
        Profile shape — one of: ``cylindrical``, ``conical``, ``exponential``,
        ``parabolic``.
    """

    D1: float  # metres
    D2: float  # metres
    A1_deg: float  # degrees (flare half-angle at input end)
    A2_deg: float  # degrees (flare half-angle at output end)
    profile_type: str = "cylindrical"

    def __post_init__(self):
        if not 0 < self.D1 < 10:
            raise ValueError(f"D1 must be a positive diameter in metres, got {self.D1}")
        if not 0 < self.D2 < 10:
            raise ValueError(f"D2 must be a positive diameter in metres, got {self.D2}")
        if self.profile_type not in (
            "cylindrical",
            "conical",
            "exponential",
            "parabolic",
        ):
            raise ValueError(
                f"profile_type must be one of cylindrical|conical|exponential|parabolic, "
                f"got {self.profile_type!r}"
            )


# ──────────────────────────────────────────────────────────────────────────────
# Helper: area / diameter conversions
# ──────────────────────────────────────────────────────────────────────────────


def _diameter_to_area(d: float) -> float:
    return math.pi * (d / 2) ** 2


def _area_to_diameter(a: float) -> float:
    return 2 * math.sqrt(a / math.pi)


# ──────────────────────────────────────────────────────────────────────────────
# Core geometry calculator
# ──────────────────────────────────────────────────────────────────────────────


def _minimum_length_conical(
    D1: float, D2: float, A1_deg: float, A2_deg: float
) -> float:
    """Minimum length for a conical adapter.

    For a constant-flare-angle cone the geometry gives::

        tan(α) = |D2 − D1| / (2 · L)

    where α is the common half-angle.  If A1 ≠ A2 we use the smaller angle
    (the more restrictive constraint) so that the adapter is feasible at both ends.

    Parameters
    ----------
    D1, D2 :
        Input and output diameters in metres.
    A1_deg, A2_deg :
        Input and output flare half-angles in degrees.

    Returns
    -------
    float
        Minimum adapter length in metres.
    """
    delta = abs(D2 - D1)
    if delta < 1e-12:
        # D1 == D2 → cylindrical geometry, no minimum length from taper
        return 0.0
    # Use the smaller angle (most restrictive constraint)
    alpha_deg = min(A1_deg, A2_deg)
    alpha_rad = math.radians(alpha_deg)
    tan_alpha = math.tan(alpha_rad)
    if tan_alpha < 1e-12:
        raise ValueError(
            f"Flare angle {alpha_deg}° is too small (tan α ≈ 0). "
            "Increase the flare angle or use a cylindrical adapter."
        )
    return delta / (2.0 * tan_alpha)


def _minimum_length_exponential(
    D1: float, D2: float, A1_deg: float, A2_deg: float
) -> float:
    """Approximate minimum length for an exponential throat adapter.

    The exponential profile A(x) = A0·exp(m·x) has a flare rate (dA/dx) that
    peaks at the small-diameter end.  To avoid an excessively sharp transition
    we constrain the local flare angle at the small end using the same conical
    criterion as a first approximation, replacing the small-end diameter with
    the mean of the two diameters.

    A more accurate bound comes from requiring the peak flare angle near the
    small end to equal A1::

        L_min ≈ |ln(Ap1/A0)| / m   where  m·A_small / sin(α) = dA/dx at x=0

    We use the simpler conical approximation with D_mean = (D1+D2)/2 as the
    effective "mean diameter" for the flare angle.  This is conservative
    (slightly over-estimates the minimum length) and matches Hornresp's
    treatment for initial design studies.
    """
    delta = abs(D2 - D1)
    if delta < 1e-12:
        return 0.0
    D_mean = (D1 + D2) / 2.0
    # Use the smaller flare angle at the small end
    alpha_deg = min(A1_deg, A2_deg)
    alpha_rad = math.radians(alpha_deg)
    tan_alpha = math.tan(alpha_rad)
    if tan_alpha < 1e-12:
        raise ValueError(
            f"Flare angle {alpha_deg}° is too small for exponential profile. "
            "Increase the flare angle."
        )
    return D_mean / (2.0 * tan_alpha)


def _minimum_length_parabolic(
    D1: float, D2: float, A1_deg: float, A2_deg: float
) -> float:
    """Approximate minimum length for a parabolic throat adapter.

    The parabolic profile uses √A(x) linear in x, which gives a gentler taper
    near the small end than the exponential.  We use the mean diameter as the
    effective conical diameter, same as the exponential approximation.
    """
    delta = abs(D2 - D1)
    if delta < 1e-12:
        return 0.0
    D_mean = (D1 + D2) / 2.0
    alpha_deg = min(A1_deg, A2_deg)
    alpha_rad = math.radians(alpha_deg)
    tan_alpha = math.tan(alpha_rad)
    if tan_alpha < 1e-12:
        raise ValueError(
            f"Flare angle {alpha_deg}° is too small for parabolic profile. "
            "Increase the flare angle."
        )
    return D_mean / (2.0 * tan_alpha)


def compute_throat_adapter(
    D1: float,
    D2: float,
    A1_deg: float,
    A2_deg: float,
    profile_type: str = "cylindrical",
    length: float | None = None,
) -> ThroatAdapter:
    """Compute a ThroatAdapter from physical design parameters.

    Parameters
    ----------
    D1 :
        Input diameter (throat chamber side) in **metres**.
    D2 :
        Output diameter (horn throat side) in **metres**.
    A1_deg :
        Input side flare half-angle in degrees.
    A2_deg :
        Output side flare half-angle in degrees.
    profile_type :
        ``cylindrical`` | ``conical`` | ``exponential`` | ``parabolic``.
    length :
        Explicit adapter length in **metres**.  If ``None``, the minimum
        geometrically feasible length is computed from the flare angles.

    Returns
    -------
    ThroatAdapter
        Dataclass ready to use in ``HornGeometry`` (fields: ``ap1``, ``lpt``,
        ``throat_adapter_type``).

    Raises
    ------
    ValueError
        If the requested length is shorter than the geometric minimum, or if
        the flare angles are too small for the given profile.

    Examples
    --------
    >>> from pyhorn_core.solver.adapter import compute_throat_adapter
    >>> # FE166NV2 throat chamber opening ~50 mm → horn throat 100 mm
    >>> adapter = compute_throat_adapter(
    ...     D1=0.050, D2=0.100, A1_deg=30.0, A2_deg=30.0, profile_type="conical"
    ... )
    >>> print(f"Lpt = {adapter.lpt*100:.2f} cm,  Ap1 = {adapter.ap1*1e4:.2f} cm²")
    Lpt = 4.33 cm,  Ap1 = 7.85 cm²
    """
    # Validate inputs
    inp = ThroatAdapterInput(
        D1=D1, D2=D2, A1_deg=A1_deg, A2_deg=A2_deg, profile_type=profile_type
    )

    # Compute output area (Ap1 = area at horn-throat end = D2)
    ap1 = _diameter_to_area(inp.D2)

    # Compute minimum length from geometry
    if profile_type == "cylindrical":
        lpt_min = 0.0
    elif profile_type == "conical":
        lpt_min = _minimum_length_conical(inp.D1, inp.D2, inp.A1_deg, inp.A2_deg)
    elif profile_type == "exponential":
        lpt_min = _minimum_length_exponential(inp.D1, inp.D2, inp.A1_deg, inp.A2_deg)
    elif profile_type == "parabolic":
        lpt_min = _minimum_length_parabolic(inp.D1, inp.D2, inp.A1_deg, inp.A2_deg)
    else:
        # Should not reach here (validated in ThroatAdapterInput)
        lpt_min = 0.0

    if length is None:
        length = lpt_min

    if length < lpt_min - 1e-12:
        raise ValueError(
            f"Requested length {length*100:.2f} cm is shorter than the "
            f"geometric minimum {lpt_min*100:.2f} cm for a {profile_type} "
            f"adapter (D1={D1*1000:.1f}mm, D2={D2*1000:.1f}mm, "
            f"A1={A1_deg}°, A2={A2_deg}°)."
        )

    return ThroatAdapter(
        type=profile_type,
        ap1=ap1,
        lpt=length,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Profile function: area as function of x along the adapter
# ──────────────────────────────────────────────────────────────────────────────


def throat_adapter_profile(
    adapter: ThroatAdapter,
    A0: float | None = None,
    n_points: int = 101,
) -> dict:
    """Area (or diameter) as a function of axial position along the adapter.

    Parameters
    ----------
    adapter :
        ``ThroatAdapter`` instance (from ``compute_throat_adapter``).
    A0 :
        Input-end area in **m²**.  If ``None``, ``A0 = adapter.ap1`` is used
        (cylindrical case), which is only correct when D1 == D2.  Caller should
        pass the throat chamber area ``atc`` for accurate results.
    n_points :
        Number of cross-sections to return.  Must be ≥ 2.

    Returns
    -------
    dict with keys
        ``x``      — 1-D numpy array of axial positions from 0 to lpt (metres)
        ``area``  — 1-D numpy array of cross-sectional areas (m²) at each x
        ``diam``   — 1-D numpy array of diameters (m) at each x
        ``A0``    — input-end area used (m²)
        ``Ap1``   — output-end area (m²), same as adapter.ap1

    The x=0 end corresponds to the throat chamber side (area A0);
    the x=lpt end corresponds to the horn throat side (area ap1).

    Example
    -------
    >>> import numpy as np
    >>> adapter = compute_throat_adapter(0.050, 0.100, 30.0, 30.0, "conical")
    >>> result = throat_adapter_profile(adapter, A0=np.pi*(0.050/2)**2, n_points=51)
    >>> print(result["x"][-1], result["area"][-1])  # should match lpt and ap1
    """
    import numpy as np

    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    lpt = adapter.lpt
    ap1 = adapter.ap1

    # Default A0: cylindrical assumption (A0 == ap1)
    if A0 is None:
        A0 = ap1

    x = np.linspace(0, lpt, n_points)

    ptype = adapter.type.lower()

    if ptype == "cylindrical" or abs(A0 - ap1) < 1e-12:
        area = np.full_like(x, ap1, dtype=float)
    elif ptype == "conical":
        # Linear taper: A(x) = A0 + (Ap1 - A0) * x / lpt
        area = A0 + (ap1 - A0) * (x / lpt)
    elif ptype == "exponential":
        # A(x) = A0 * exp(m * x),  m = ln(Ap1/A0) / lpt
        m = math.log(ap1 / A0) / lpt
        area = A0 * np.exp(m * x)
    elif ptype == "parabolic":
        # √A(x) is linear: √A(x) = √A0 + (√Ap1 - √A0) * x/lpt
        sq_A0 = math.sqrt(A0)
        sq_ap1 = math.sqrt(ap1)
        sq_area = sq_A0 + (sq_ap1 - sq_A0) * (x / lpt)
        area = sq_area**2
    else:
        # Unknown type — return cylindrical
        area = np.full_like(x, ap1, dtype=float)

    diam = _area_to_diameter_array(area)

    return {
        "x": x,
        "area": area,
        "diam": diam,
        "A0": A0,
        "Ap1": ap1,
    }


def _area_to_diameter_array(area_array) -> "ndarray":
    """Vectorised diameter from area array."""
    import numpy as np

    return 2.0 * np.sqrt(area_array / math.pi)
