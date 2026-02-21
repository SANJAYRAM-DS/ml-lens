# ==============================
# File: linalg/algorithms/decomposition/inverse.py
# ==============================
"""Matrix inverse via Gauss-Jordan elimination."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg._internal.constants import SINGULAR_PIVOT_THRESHOLD
from mllense.math.linalg.algorithms.decomposition.base import BaseDecomposition
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.core.validation import validate_square
from mllense.math.linalg.exceptions import SingularMatrixError

__all__ = ["Inverse"]


class Inverse(BaseDecomposition):
    """Compute the matrix inverse via Gauss-Jordan elimination."""

    metadata = AlgorithmMetadata(
        name="inverse",
        operation="inverse",
        complexity="O(n^3)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Matrix inverse via Gauss-Jordan row reduction.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        a: InternalMatrix = args[0]
        n = validate_square(a, operation="inverse")

        trace.record(
            operation="inverse_start",
            description=f"Computing inverse of {n}×{n} matrix",
        )

        # augment with identity
        aug: list[list[float]] = [
            [a[i][j] for j in range(n)] + [1.0 if i == k else 0.0 for k in range(n)]
            for i in range(n)
        ]

        for col in range(n):
            # partial pivot
            max_val = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row

            if max_val < SINGULAR_PIVOT_THRESHOLD:
                raise SingularMatrixError(
                    f"Zero pivot at column {col}. Matrix is singular."
                )

            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            for j in range(2 * n):
                aug[col][j] /= pivot

            for row in range(n):
                if row == col:
                    continue
                factor = aug[row][col]
                for j in range(2 * n):
                    aug[row][j] -= factor * aug[col][j]

        result = [[aug[i][j + n] for j in range(n)] for i in range(n)]

        trace.record(
            operation="inverse_done",
            description=f"Inverse computed for {n}×{n} matrix",
            data=result,
        )

        return result
