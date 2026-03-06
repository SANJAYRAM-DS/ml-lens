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

        what = context.what_lense_enabled
        how = context.how_lense_enabled

        if what:
            self.what_lense = (
                "=== WHAT: Matrix Multiplication ===\n"
                "Matrix multiplication (dot product) combines the rows of the first matrix with the "
                "columns of the second. Each element in the result is the sum of the products of "
                "corresponding elements.\n\n"
                "=== WHY we need it in ML ===\n"
                "It allows us to compute many linear combinations at once. It forms the core of "
                "feed-forward neural networks (Weights * Inputs), attention mechanisms (Q * K^T), "
                "and embedding projections.\n\n"
                "=== WHERE it is used in Real ML ===\n"
                "1. Dense/Linear Layers: Output = Weights @ Inputs + Bias.\n"
                "2. Convolutions: Often lowered to matrix multiplication (im2col).\n"
                "3. Transformers: Self-attention relies heavily on batched matrix multiplications."
            )
        else:
            self.what_lense = ""

        if how:
            checkpoints = []
            checkpoints.append(f"1. Validated shapes: A({m}x{k}) @ B({k}x{n}) -> Result({m}x{n}).")
            checkpoints.append(f"2. Initialized empty output matrix of shape {m}x{n}.")
            checkpoints.append(f"3. Triple-loop computation (m={m}, k={k}, n={n}):")
            
            total_ops = m * k * n
            op_count = 0
            omitted = False

            for i in range(m):
                for j in range(n):
                    # For each cell in result, we did k products
                    for j_k in range(k):
                        op_count += 1
                        if total_ops <= 10 or op_count <= 5 or op_count > total_ops - 5:
                            val_a = a[i][j_k]
                            val_b = b[j_k][j]
                            checkpoints.append(
                                f"   - Result[{i}][{j}] += A[{i}][{j_k}] * B[{j_k}][{j}] "
                                f"({val_a} * {val_b} = {val_a * val_b})"
                            )
                        elif not omitted:
                            checkpoints.append("   - ... (skipped intermediate multiplications) ...")
                            omitted = True

            checkpoints.append("4. Finished matrix multiplication.")
            self.how_lense = "\n".join(checkpoints)
        else:
            self.how_lense = ""

        # collapse to vector or scalar when appropriate
        if m == 1 and n == 1:
            return result[0][0]
        if n == 1:
            return [row[0] for row in result]
        if m == 1:
            return result[0]
        return result
