"""Shared horn-design builders used by optimization layers."""

from typing import Any, Mapping, Optional

from pyhorn_core.config.models import HornGeometry


def build_horn_from_params(
    params: Mapping[str, Any],
    profile_type: Optional[str] = None,
    enclosure_type: Optional[str] = None,
    n_segments: Optional[int] = None,
) -> HornGeometry:
    """Construct a HornGeometry from optimizer-style parameter mappings."""
    throat_area = float(params["throat_area"])
    lrc = float(params.get("lrc", 0.0))
    resolved_profile = profile_type or params.get("profile_type")
    if resolved_profile is None:
        raise ValueError("profile_type is required to build horn geometry")

    resolved_enclosure = enclosure_type or params.get("enclosure_type", "BLH")
    resolved_segments = int(
        n_segments if n_segments is not None else params.get("n_segments", 100)
    )

    return HornGeometry(
        throat_area=throat_area,
        mouth_area=float(params["mouth_area"]),
        path_length=float(params["path_length"]),
        profile_type=str(resolved_profile),
        hyperbolic_t=float(params.get("hyperbolic_t", 1.0)),
        n_segments=resolved_segments,
        enclosure_type=str(resolved_enclosure),
        lrc=lrc,
        vrc=float(params.get("vrc", lrc * throat_area)),
        vtc=float(params.get("vtc", 0.0)),
    )
