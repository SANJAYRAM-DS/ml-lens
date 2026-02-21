# ==============================
# File: linalg/_internal/decorators.py
# ==============================
"""Internal decorators used across the linalg engine.

These are NOT part of the public API.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def algorithm(func: F) -> F:
    """Mark a function as an algorithm entry point.

    Currently just passes through; reserved for future instrumentation
    (e.g. automatic trace injection, profiling hooks).
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)
    return wrapper  # type: ignore[return-value]


def register_backend(name: str) -> Callable[[type], type]:
    """Class decorator that registers a backend in the backend registry.

    Usage::

        @register_backend("python")
        class PythonBackend(Backend):
            ...
    """
    def decorator(cls: type) -> type:
        from mllense.math.linalg.registry.backend_registry import backend_registry
        backend_registry.register(name, cls)
        return cls
    return decorator


def timed(func: F) -> F:
    """Decorator that measures and prints execution time."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timed] {func.__qualname__}: {elapsed:.6f}s")
        return result
    return wrapper  # type: ignore[return-value]


def validate_inputs(*validators: Callable[..., None]) -> Callable[[F], F]:
    """Decorator that runs a sequence of validator functions before the main call.

    Each validator receives the same ``*args, **kwargs`` as the decorated
    function and should raise on invalid input.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for v in validators:
                v(*args, **kwargs)
            return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return decorator