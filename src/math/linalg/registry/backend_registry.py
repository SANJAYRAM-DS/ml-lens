# ==============================
# File: linalg/registry/backend_registry.py
# ==============================
"""Central registry for compute backends.

Backends are registered by name and retrieved at runtime.
There is no hardcoded global state — only explicit registration.
"""

from __future__ import annotations

from typing import Dict, Type

from mllense.math.linalg.exceptions import InvalidBackendError

__all__ = ["BackendRegistry", "backend_registry"]


class BackendRegistry:
    """Singleton-style registry mapping backend names to backend classes.

    Usage::

        from mllense.math.linalg.registry.backend_registry import backend_registry

        backend_registry.register("numpy", NumpyBackend)
        backend = backend_registry.get("numpy")   # returns an *instance*
    """

    def __init__(self) -> None:
        self._backends: Dict[str, Type] = {}
        self._instances: Dict[str, object] = {}

    def register(self, name: str, backend_cls: Type) -> None:
        """Register a backend class under *name* (case-insensitive)."""
        key = name.strip().lower()
        self._backends[key] = backend_cls
        # invalidate cached instance so next .get() creates a fresh one
        self._instances.pop(key, None)

    def get(self, name: str) -> object:
        """Return a (cached) backend instance for *name*.

        Raises:
            InvalidBackendError: If the name is not registered.
        """
        key = name.strip().lower()
        if key not in self._backends:
            raise InvalidBackendError(key)
        if key not in self._instances:
            self._instances[key] = self._backends[key]()
        return self._instances[key]

    def available(self) -> list[str]:
        """Return sorted list of registered backend names."""
        return sorted(self._backends.keys())

    def is_registered(self, name: str) -> bool:
        return name.strip().lower() in self._backends

    def clear(self) -> None:
        """Remove all registrations (useful for testing)."""
        self._backends.clear()
        self._instances.clear()

    def __repr__(self) -> str:
        return f"BackendRegistry(backends={self.available()})"


# Module-level singleton
backend_registry = BackendRegistry()
