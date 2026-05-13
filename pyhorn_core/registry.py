"""Backward-compat shim — registry moved to pyhorn_registry.

All code should now import from pyhorn_registry directly:
    from pyhorn_registry import Registry, RegistryEntry, registry

This shim is kept for external code that still imports from pyhorn_core.registry.
"""

try:
    from pyhorn_registry import (  # noqa: F401
        REGISTRY_FILENAME,
        REGISTRY_VERSION,
        Registry,
        RegistryEntry,
        registry,
    )
    _HAS_PYHORN_REGISTRY = True
except ImportError:
    REGISTRY_FILENAME = None
    REGISTRY_VERSION = None
    Registry = None
    RegistryEntry = None
    registry = None
