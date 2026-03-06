# ==============================
# File: linalg/algorithms/elementwise/scalar.py
# ==============================
"""Scalar-matrix operations (scalar * matrix, matrix + scalar, etc.)."""

from __future__ import annotations

from typing import Any, Union

from mllense.math.linalg.algorithms.elementwise.base import BaseElementwise
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import EmptyMatrixError

__all__ = ["ScalarMultiply", "ScalarAdd"]


class ScalarMultiply(BaseElementwise):
    """Multiply every element by a scalar: ``C[i][j] = scalar * A[i][j]``."""

    metadata = AlgorithmMetadata(
        name="scalar_multiply",
        operation="scalar_multiply",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Scalar multiplication of a matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        """Args: args[0] = matrix, args[1] = scalar."""
        m: InternalMatrix = args[0]
        scalar: float = float(args[1])

        rows = len(m)
        if rows == 0:
            raise EmptyMatrixError("Cannot scale an empty matrix.")
        cols = len(m[0])

        trace.record(
            operation="scalar_multiply",
            description=f"Scaling {rows}×{cols} matrix by {scalar}",
        )

        return [[v * scalar for v in row] for row in m]


class ScalarAdd(BaseElementwise):
    """Add a scalar to every element: ``C[i][j] = A[i][j] + scalar``."""

    metadata = AlgorithmMetadata(
        name="scalar_add",
        operation="scalar_add",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Add a scalar to every element of a matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        m: InternalMatrix = args[0]
        scalar: float = float(args[1])

        rows = len(m)
        if rows == 0:
            raise EmptyMatrixError("Cannot add to an empty matrix.")
        cols = len(m[0])

        trace.record(
            operation="scalar_add",
            description=f"Adding {scalar} to {rows}×{cols} matrix",
        )

        return [[v + scalar for v in row] for row in m]
