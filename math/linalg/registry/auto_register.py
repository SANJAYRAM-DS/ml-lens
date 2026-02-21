# ==============================
# File: linalg/registry/auto_register.py
# ==============================
"""Auto-registration of backends and algorithms.

Called once during package initialisation.  Uses explicit imports
(no dangerous ``importlib`` scanning) to keep the dependency graph
predictable and debuggable.
"""

from __future__ import annotations

__all__ = ["auto_register"]

_REGISTERED: bool = False


def auto_register() -> None:
    """Register all built-in backends and algorithms.

    This function is idempotent — calling it more than once is a no-op.
    """
    global _REGISTERED
    if _REGISTERED:
        return

    _register_backends()
    _register_algorithms()
    _REGISTERED = True


def _register_backends() -> None:
    from mllense.math.linalg.backend.numpy_backend import NumpyBackend
    from mllense.math.linalg.backend.python_backend import PythonBackend
    from mllense.math.linalg.registry.backend_registry import backend_registry

    backend_registry.register("python", PythonBackend)
    backend_registry.register("numpy", NumpyBackend)


def _register_algorithms() -> None:
    from mllense.math.linalg.algorithms.matmul.naive import NaiveMatmul
    from mllense.math.linalg.algorithms.solve.gaussian import GaussianSolve
    from mllense.math.linalg.registry.algorithm_registry import algorithm_registry

    algorithm_registry.register("matmul", "naive", NaiveMatmul, default=True)
    algorithm_registry.register("solve", "gaussian", GaussianSolve, default=True)
