# ==============================
# File: linalg/algorithms/solve/cholesky.py
# ==============================
"""Cholesky decomposition and Cholesky-based solve for SPD matrices.

Only applicable to symmetric positive-definite (SPD) matrices.
Complexity: O(n³/3).
"""

from __future__ import annotations

import math
from typing import Any, Tuple

from mllense.math.linalg.algorithms.solve.base import BaseSolve
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_square
from mllense.math.linalg.exceptions import InvalidInputError, SingularMatrixError

__all__ = ["CholeskySolve", "cholesky_decompose"]


def cholesky_decompose(
    a: InternalMatrix, trace: Trace | None = None
) -> InternalMatrix:
    """Compute the Cholesky decomposition ``A = L L^T``.

    Args:
        a: Symmetric positive-definite matrix (n×n).

    Returns:
        Lower-triangular matrix L such that ``A = L @ L^T``.
    """
    n = len(a)
    l: InternalMatrix = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1):
            s = math.fsum(l[i][k] * l[j][k] for k in range(j))
            if i == j:
                val = a[i][i] - s
                if val <= 0.0:
                    raise InvalidInputError(
                        f"Matrix is not positive-definite: "
                        f"a[{i}][{i}] - sum = {val:.2e} <= 0"
                    )
                l[i][j] = math.sqrt(val)
            else:
                if abs(l[j][j]) < 1e-15:
                    raise SingularMatrixError(
                        f"Zero diagonal L[{j}][{j}] during Cholesky."
                    )
                l[i][j] = (a[i][j] - s) / l[j][j]

    if trace is not None:
        trace.record(
            operation="cholesky_decompose",
            description=f"Cholesky decomposition complete for {n}×{n} SPD matrix",
            data=l,
        )

    return l


class CholeskySolve(BaseSolve):
    """Solve ``Ax = b`` using Cholesky decomposition for SPD matrices."""

    metadata = AlgorithmMetadata(
        name="cholesky_solve",
        operation="solve",
        complexity="O(n^3/3)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Cholesky decomposition solve for symmetric positive-definite systems.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalVector:
        a: InternalMatrix = args[0]
        b: InternalVector = args[1]
        n = validate_square(a, operation="cholesky_solve")

        trace.record(
            operation="cholesky_solve_start",
            description=f"Cholesky solve on {n}×{n} system",
        )

        l = cholesky_decompose(a, trace=trace)

        # forward substitution: Ly = b
        y: InternalVector = [0.0] * n
        for i in range(n):
            s = math.fsum(l[i][j] * y[j] for j in range(i))
            y[i] = (b[i] - s) / l[i][i]

        # back substitution: L^T x = y
        x: InternalVector = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = math.fsum(l[j][i] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - s) / l[i][i]

        trace.record(
            operation="cholesky_solve_done",
            description=f"Solution vector (length {n})",
            data=x,
        )

        return x
