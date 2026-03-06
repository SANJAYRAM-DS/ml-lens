# ==============================
# File: linalg/algorithms/elementwise/subtract.py
# ==============================
"""Element-wise matrix subtraction."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.elementwise.base import BaseElementwise
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import ShapeMismatchError, EmptyMatrixError

__all__ = ["ElementwiseSubtract"]


class ElementwiseSubtract(BaseElementwise):
    """Element-wise subtraction: ``C[i][j] = A[i][j] - B[i][j]``."""

    metadata = AlgorithmMetadata(
        name="elementwise_subtract",
        operation="subtract",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Element-wise matrix subtraction.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        a: InternalMatrix = args[0]
        b: InternalMatrix = args[1]

        a_rows = len(a)
        a_cols = len(a[0]) if a_rows else 0
        b_rows = len(b)
        b_cols = len(b[0]) if b_rows else 0

        if a_rows == 0 or a_cols == 0:
            raise EmptyMatrixError("Cannot subtract empty matrices.")

        if a_rows != b_rows or a_cols != b_cols:
            raise ShapeMismatchError(
                expected=f"same shape ({a_rows}×{a_cols})",
                got=f"({b_rows}×{b_cols})",
                operation="subtract",
            )

        trace.record(
            operation="elementwise_subtract",
            description=f"Subtracting {a_rows}×{a_cols} matrices",
        )

        return [
            [a[i][j] - b[i][j] for j in range(a_cols)]
            for i in range(a_rows)
        ]
