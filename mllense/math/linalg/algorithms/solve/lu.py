# ==============================
# File: linalg/algorithms/solve/lu.py
# ==============================
"""LU Decomposition with partial pivoting, and LU-based solve."""

from __future__ import annotations

import math
from typing import Any, Tuple

from mllense.math.linalg._internal.constants import SINGULAR_PIVOT_THRESHOLD
from mllense.math.linalg.algorithms.solve.base import BaseSolve
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_square
from mllense.math.linalg.exceptions import SingularMatrixError

__all__ = ["LUSolve", "lu_decompose"]


def lu_decompose(
    a: InternalMatrix, trace: Trace | None = None
) -> Tuple[InternalMatrix, InternalMatrix, list[int]]:
    """Compute PA = LU decomposition with partial pivoting.

    Returns:
        (L, U, perm) where ``perm`` is the row permutation vector.
    """
    n = len(a)
    # deep copy
    u: InternalMatrix = [row[:] for row in a]
    l: InternalMatrix = [[0.0] * n for _ in range(n)]
    perm = list(range(n))

    for col in range(n):
        # partial pivot
        max_val = abs(u[col][col])
        max_row = col
        for row in range(col + 1, n):
            if abs(u[row][col]) > max_val:
                max_val = abs(u[row][col])
                max_row = row

        if max_val < SINGULAR_PIVOT_THRESHOLD:
            raise SingularMatrixError(
                f"Near-zero pivot at column {col} during LU decomposition."
            )

        if max_row != col:
            u[col], u[max_row] = u[max_row], u[col]
            l[col], l[max_row] = l[max_row], l[col]
            perm[col], perm[max_row] = perm[max_row], perm[col]

        l[col][col] = 1.0

        for row in range(col + 1, n):
            factor = u[row][col] / u[col][col]
            l[row][col] = factor
            for j in range(col, n):
                u[row][j] -= factor * u[col][j]
            u[row][col] = 0.0

    # set remaining diagonal of L
    for i in range(n):
        l[i][i] = 1.0

    if trace is not None:
        trace.record(
            operation="lu_decompose",
            description=f"LU decomposition complete for {n}×{n} matrix",
            data={"L": l, "U": u, "perm": perm},
        )

    return l, u, perm


class LUSolve(BaseSolve):
    """Solve ``Ax = b`` via LU decomposition with partial pivoting."""

    metadata = AlgorithmMetadata(
        name="lu_solve",
        operation="solve",
        complexity="O(n^3)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Solve Ax = b via LU decomposition with partial pivoting.",
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
        n = validate_square(a, operation="lu_solve")

        trace.record(
            operation="lu_solve_start",
            description=f"LU solve on {n}×{n} system",
        )

        l, u, perm = lu_decompose(a, trace=trace)

        # apply permutation to b
        pb = [b[perm[i]] for i in range(n)]

        # forward substitution: Ly = Pb
        y: InternalVector = [0.0] * n
        for i in range(n):
            s = math.fsum(l[i][j] * y[j] for j in range(i))
            y[i] = pb[i] - s

        # back substitution: Ux = y
        x: InternalVector = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = math.fsum(u[i][j] * x[j] for j in range(i + 1, n))
            if abs(u[i][i]) < SINGULAR_PIVOT_THRESHOLD:
                raise SingularMatrixError(
                    f"Zero diagonal at U[{i}][{i}] during back-substitution."
                )
            x[i] = (y[i] - s) / u[i][i]

        trace.record(
            operation="lu_solve_done",
            description=f"Solution vector (length {n})",
            data=x,
        )

        return x
