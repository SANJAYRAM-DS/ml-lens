# ==============================
# File: linalg/core/execution_context.py
# ==============================
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

    # -- immutable properties --------------------------------------------- #
    @property
    def backend(self) -> str:
        return self._backend  # type: ignore[return-value]

    @property
    def mode(self) -> ExecutionMode:
        return self._mode  # type: ignore[return-value]

    @property
    def trace_enabled(self) -> bool:
        return self._trace_enabled  # type: ignore[return-value]

    @property
    def algorithm_hint(self) -> str | None:
        return self._algorithm_hint  # type: ignore[return-value]

    @property
    def what_lense_enabled(self) -> bool:
        return self._what_lense_enabled  # type: ignore[return-value]

    @property
    def how_lense_enabled(self) -> bool:
        return self._how_lense_enabled  # type: ignore[return-value]

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError(
            f"ExecutionContext is immutable — cannot set '{key}'."
        )

    def __delattr__(self, key: str) -> None:
        raise AttributeError(
            f"ExecutionContext is immutable — cannot delete '{key}'."
        )

    # -- helpers ---------------------------------------------------------- #
    def with_overrides(self, **kwargs: Any) -> ExecutionContext:
        """Return a *new* context with selected fields overridden."""
        return ExecutionContext(
            backend=kwargs.get("backend", self._backend),  # type: ignore[arg-type]
            mode=kwargs.get("mode", self._mode),  # type: ignore[arg-type]
            trace_enabled=kwargs.get("trace_enabled", self._trace_enabled),  # type: ignore[arg-type]
            algorithm_hint=kwargs.get("algorithm_hint", self._algorithm_hint),  # type: ignore[arg-type]
            what_lense_enabled=kwargs.get("what_lense_enabled", self._what_lense_enabled),  # type: ignore[arg-type]
            how_lense_enabled=kwargs.get("how_lense_enabled", self._how_lense_enabled),  # type: ignore[arg-type]
        )

    @staticmethod
    def from_config() -> ExecutionContext:
        """Build a context from the current :class:`GlobalConfig` singleton."""
        from mllense.math.linalg.config import get_config

        cfg = get_config()
        return ExecutionContext(
            backend=cfg.default_backend,
            mode=ExecutionMode.from_string(cfg.default_mode),
            trace_enabled=cfg.trace_enabled,
        )

    def __repr__(self) -> str:
        return (
            f"ExecutionContext(backend={self.backend!r}, mode={self.mode!r}, "
            f"trace_enabled={self.trace_enabled!r}, "
            f"algorithm_hint={self.algorithm_hint!r}, "
            f"what_lense_enabled={self.what_lense_enabled!r}, "
            f"how_lense_enabled={self.how_lense_enabled!r})"
        )
