"""Acoustic solver module for pyhorn_core."""

from pyhorn_core.solver import (  # noqa: F401
    adapter,
    design,
    geometry_discretise,
    hornresp,
    hornresp_parser,
    # models, optimizer: moved to pyhorn_physics/orchestrators.py.
    # Re-export shim kept at solver/models.py for backward compat.
    # Import directly from pyhorn_physics.orchestrators instead.
    profiles,
    scoring,
    spectrogram,
    time_domain,
)

# medial_axis intentionally omitted here — it imports pyhorn_segment which is
# an optional dependency. Commands that need it import it lazily (auto-segment).
