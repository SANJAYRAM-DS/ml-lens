"""Immutable execution context threaded through every computation."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.core.mode import ExecutionMode

__all__ = ["ExecutionContext"]


class ExecutionContext:
    """Immutable container that carries per-call execution parameters.

    Parameters:
        backend: Name of the backend to use (e.g. ``"numpy"``, ``"python"``).
        mode: The :class:`ExecutionMode` for this call.
        trace_enabled: Whether to record intermediate steps.
        algorithm_hint: Optional algorithm name override (bypass auto-selection).
    """

    __slots__ = (
        "_backend",
        "_mode",
        "_trace_enabled",
        "_algorithm_hint",
        "_what_lense_enabled",
        "_how_lense_enabled",
    )

    def __init__(
        self,
        backend: str,
        mode: ExecutionMode,
        trace_enabled: bool = False,
        algorithm_hint: str | None = None,
        what_lense_enabled: bool = True,
        how_lense_enabled: bool = False,
    ) -> None:
        from mllense.math.linalg.registry.backend_registry import backend_registry
        from mllense.math.linalg.exceptions import InvalidBackendError
        
        if not backend_registry.is_registered(backend):
            raise InvalidBackendError(backend)
            
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_trace_enabled", trace_enabled)
        object.__setattr__(self, "_algorithm_hint", algorithm_hint)
        object.__setattr__(self, "_what_lense_enabled", what_lense_enabled)
        object.__setattr__(self, "_how_lense_enabled", how_lense_enabled)
        
    @property
    def backend(self) -> str:
        return self._backend
    
    @property
    def mode(self) -> ExecutionMode:
        return self._backend
    
    @property
    def trace_enabled(self) -> ExecutionMode:
        return self._trace_enabled
    
    @property
    def 