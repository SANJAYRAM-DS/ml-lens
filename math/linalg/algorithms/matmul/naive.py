# ==============================
# File: linalg/algorithms/matmul/naive.py
# ==============================
"""Naive (triple-loop) matrix multiplication algorithm.

Complexity: O(m * k * n)  —  cache-unfriendly but simple and correct.

Supports:
    - 2D × 2D  → 2D
    - 2D × 1D  → 1D  (column vector on right)
    - 1D × 1D  → scalar (inner product)

Edge-case handling:
    - Empty matrices  → EmptyMatrixError
    - Shape mismatch  → ShapeMismatchError
    - Float overflow   → NumericalInstabilityError
"""

from __future__ import annotations

import math
from typing import Any, Union

from mllense.math.linalg._internal.constants import FLOAT_OVERFLOW_GUARD
from mllense.math.linalg.algorithms.base import BaseAlgorithm
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_matmul_shapes
from mllense.math.linalg.exceptions import (
    EmptyMatrixError,
    NumericalInstabilityError,
    ShapeMismatchError,
)

__all__ = ["NaiveMatmul"]


class NaiveMatmul(BaseAlgorithm):
    """Triple-loop matrix multiplication: ``C[i][j] = Σ_k A[i][k] * B[k][j]``."""

    metadata = AlgorithmMetadata(
        name="naive_matmul",
        operation="matmul",
        complexity="O(m*k*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description=(
            "Naive triple-loop matrix multiplication.  "
            "Simple, correct, but O(n³) with poor cache locality."
        ),
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Union[InternalMatrix, InternalVector, float]:
        """Multiply two matrices / vectors.

        Positional args:
            args[0]: A  (InternalMatrix — 2-D list)
            args[1]: B  (InternalMatrix — 2-D list)

        Both operands are already validated and in internal format
        by the API layer.  The shapes must be compatible:
        ``A.cols == B.rows``.

        Returns:
            InternalMatrix, InternalVector, or float depending on
            the shapes of A and B.
        """
        a: InternalMatrix = args[0]
        b: InternalMatrix = args[1]

        a_rows = len(a)
        a_cols = len(a[0]) if a_rows else 0
        b_rows = len(b)
        b_cols = len(b[0]) if b_rows else 0

        # shape validation
        m, k, n = validate_matmul_shapes((a_rows, a_cols), (b_rows, b_cols))

        trace.record(
            operation="matmul_start",
            description=f"Naive matmul: ({m}×{k}) @ ({k}×{n})",
            complexity_note=f"O({m}*{k}*{n}) = O({m * k * n}) multiplications",
        )

        # allocate result
        result: InternalMatrix = [[0.0] * n for _ in range(m)]

        for i in range(m):
            for j_k in range(k):
                a_val = a[i][j_k]
                if a_val == 0.0:
                    continue
                for j in range(n):
                    result[i][j] += a_val * b[j_k][j]

                    # overflow guard
                    if abs(result[i][j]) > FLOAT_OVERFLOW_GUARD:
                        raise NumericalInstabilityError(
                            f"Float overflow at result[{i}][{j}] = {result[i][j]}"
                        )

        trace.record(
            operation="matmul_done",
            description=f"Result shape: ({m}×{n})",
            data=result,
        )

        # collapse to vector or scalar when appropriate
        if m == 1 and n == 1:
            return result[0][0]
        if n == 1:
            return [row[0] for row in result]
        if m == 1:
            return result[0]
        return result
