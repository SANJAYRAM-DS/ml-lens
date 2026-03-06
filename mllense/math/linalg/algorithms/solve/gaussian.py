# ==============================
# File: linalg/algorithms/solve/gaussian.py
# ==============================
"""Gaussian Elimination with partial pivoting for solving Ax = b.

Complexity: O(n³)

Supports:
    - Square coefficient matrix A
    - Single right-hand-side vector b
    - Partial pivoting for numerical stability

Edge-case handling:
    - Singular / near-singular matrices → SingularMatrixError
    - Division by zero                  → caught by pivot threshold
    - Float overflow                     → NumericalInstabilityError
    - Empty systems                      → EmptyMatrixError
"""

from __future__ import annotations

import math
from typing import Any

from mllense.math.linalg._internal.constants import (
    FLOAT_OVERFLOW_GUARD,
    SINGULAR_PIVOT_THRESHOLD,
)
from mllense.math.linalg.algorithms.base import BaseAlgorithm
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_solve_shapes
from mllense.math.linalg.exceptions import (
    NumericalInstabilityError,
    SingularMatrixError,
)

__all__ = ["GaussianSolve"]


class GaussianSolve(BaseAlgorithm):
    """Solve ``Ax = b`` using Gaussian elimination with partial pivoting."""

    metadata = AlgorithmMetadata(
        name="gaussian_elimination",
        operation="solve",
        complexity="O(n^3)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description=(
            "Gaussian elimination with partial pivoting.  "
            "Forward-eliminates to upper-triangular form, then back-substitutes."
        ),
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalVector:
        """Solve ``Ax = b``.

        Positional args:
            args[0]: A  (InternalMatrix — n×n)
            args[1]: b  (InternalVector — length n)

        Returns:
            x  (InternalVector — length n)
        """
        a: InternalMatrix = args[0]
        b: InternalVector = args[1]

        n = validate_solve_shapes(a, b)

        trace.record(
            operation="gaussian_start",
            description=f"Gaussian elimination on {n}×{n} system",
            complexity_note=f"O({n}^3) = O({n ** 3}) operations",
        )

        # build augmented matrix [A | b]  — deep-copy to avoid mutation
        aug: list[list[float]] = [
            [a[i][j] for j in range(n)] + [b[i]] for i in range(n)
        ]

        # ── forward elimination with partial pivoting ─────────────────── #
        for col in range(n):
            # find row with largest absolute value in this column
            max_abs = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, n):
                val = abs(aug[row][col])
                if val > max_abs:
                    max_abs = val
                    max_row = row

            # pivot check
            if max_abs < SINGULAR_PIVOT_THRESHOLD:
                raise SingularMatrixError(
                    f"Near-zero pivot ({max_abs:.2e}) at column {col}. "
                    f"Matrix is singular or nearly singular."
                )

            # swap rows if needed
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]
                trace.record(
                    operation="pivot_swap",
                    description=f"Swapped rows {col} and {max_row}",
                    data={"col": col, "swapped_rows": (col, max_row)},
                )

            pivot_val = aug[col][col]

            # eliminate below
            for row in range(col + 1, n):
                factor = aug[row][col] / pivot_val
                aug[row][col] = 0.0  # exact zero
                for j in range(col + 1, n + 1):
                    aug[row][j] -= factor * aug[col][j]
                    # overflow guard
                    if abs(aug[row][j]) > FLOAT_OVERFLOW_GUARD:
                        raise NumericalInstabilityError(
                            f"Float overflow during elimination at "
                            f"aug[{row}][{j}] = {aug[row][j]}"
                        )

            trace.record(
                operation="elimination_step",
                description=f"Eliminated column {col}",
                data=[row[:] for row in aug],  # snapshot
            )

        # ── back substitution ─────────────────────────────────────────── #
        x: InternalVector = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = math.fsum(aug[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (aug[i][n] - s) / aug[i][i]

            if abs(x[i]) > FLOAT_OVERFLOW_GUARD:
                raise NumericalInstabilityError(
                    f"Float overflow in back-substitution: x[{i}] = {x[i]}"
                )

        trace.record(
            operation="gaussian_done",
            description=f"Solution vector computed (length {n})",
            data=x,
        )

        return x
