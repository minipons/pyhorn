"""Backward-compat shim — registry moved to pyhorn_registry.

All code should now import from pyhorn_registry directly:
    from pyhorn_registry import Registry, RegistryEntry, registry

This shim is kept for external code that still imports from pyhorn_core.registry.
"""

from pyhorn_registry import (  # noqa: F401
    REGISTRY_FILENAME,
    REGISTRY_VERSION,
    Registry,
    RegistryEntry,
    registry,
)
