# ==============================
# File: linalg/registry/algorithm_registry.py
# ==============================
"""Central registry for algorithms.

Algorithms are grouped by *operation* (e.g. ``"matmul"``, ``"solve"``)
and within each operation by *algorithm name* (e.g. ``"naive"``, ``"gaussian"``).

Auto-selection logic lives here as well.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Type

from mllense.math.linalg._internal.constants import (
    MEDIUM_MATRIX_THRESHOLD,
    SMALL_MATRIX_THRESHOLD,
)
from mllense.math.linalg.exceptions import AlgorithmNotFoundError

if TYPE_CHECKING:
    from mllense.math.linalg.algorithms.base import BaseAlgorithm
    from mllense.math.linalg.core.execution_context import ExecutionContext

__all__ = ["AlgorithmRegistry", "algorithm_registry"]


class AlgorithmRegistry:
    """Singleton-style registry for algorithm classes.

    Structure::

        {
            "matmul": {"naive": NaiveMatmul, ...},
            "solve":  {"gaussian": GaussianSolve, ...},
        }
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Dict[str, Type[BaseAlgorithm]]] = {}
        self._defaults: Dict[str, str] = {}

    # ── registration ──────────────────────────────────────────────────── #

    def register(
        self,
        operation: str,
        name: str,
        algorithm_cls: Type[BaseAlgorithm],
        *,
        default: bool = False,
    ) -> None:
        """Register an algorithm class.

        Args:
            operation: Operation family (e.g. ``"matmul"``).
            name: Unique algorithm name within the family (e.g. ``"naive"``).
            algorithm_cls: The algorithm class (must subclass ``BaseAlgorithm``).
            default: If ``True``, set as default for this operation.
        """
        op = operation.strip().lower()
        alg_name = name.strip().lower()
        if op not in self._registry:
            self._registry[op] = {}
        self._registry[op][alg_name] = algorithm_cls
        if default or op not in self._defaults:
            self._defaults[op] = alg_name

    # ── retrieval ─────────────────────────────────────────────────────── #

    def get(
        self,
        operation: str,
        context: ExecutionContext,
        matrix_dim: int | None = None,
    ) -> BaseAlgorithm:
        """Get an algorithm **instance** for the given operation and context.

        Selection priority:
        1. ``context.algorithm_hint`` if provided.
        2. Auto-select based on backend and matrix size.
        3. Registered default for this operation.

        Returns:
            An instantiated algorithm.

        Raises:
            AlgorithmNotFoundError: If no algorithm can be resolved.
        """
        op = operation.strip().lower()

        if op not in self._registry or not self._registry[op]:
            raise AlgorithmNotFoundError(op)

        # 1. explicit hint
        if context.algorithm_hint:
            alg_name = context.algorithm_hint.strip().lower()
            if alg_name not in self._registry[op]:
                raise AlgorithmNotFoundError(op, alg_name)
            return self._registry[op][alg_name]()

        # 2. auto-selection
        alg_name = self._auto_select(op, context, matrix_dim)
        return self._registry[op][alg_name]()

    def _auto_select(
        self,
        operation: str,
        context: ExecutionContext,
        matrix_dim: int | None,
    ) -> str:
        """Deterministic auto-selection logic.

        * ``numpy`` backend → always delegate to the ``"numpy_delegate"``
          variant if registered, else fall back to default.
        * ``python`` backend → pick by matrix size:
            - small → ``"naive"`` (if available)
            - medium/large → ``"block"`` (if available), else ``"naive"``
        """
        available = self._registry.get(operation, {})

        # backend-aware shortcut
        if context.backend == "numpy" and "numpy_delegate" in available:
            return "numpy_delegate"

        # size-aware fallback for python backend
        if matrix_dim is not None and matrix_dim > SMALL_MATRIX_THRESHOLD:
            if "block" in available:
                return "block"

        # default
        return self._defaults.get(operation, next(iter(available)))

    # ── introspection ─────────────────────────────────────────────────── #

    def list_operations(self) -> list[str]:
        return sorted(self._registry.keys())

    def list_algorithms(self, operation: str) -> list[str]:
        op = operation.strip().lower()
        if op not in self._registry:
            return []
        return sorted(self._registry[op].keys())

    def clear(self) -> None:
        """Remove all registrations."""
        self._registry.clear()
        self._defaults.clear()

    def __repr__(self) -> str:
        return f"AlgorithmRegistry(operations={self.list_operations()})"


# Module-level singleton
algorithm_registry = AlgorithmRegistry()
