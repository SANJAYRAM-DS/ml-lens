# ==============================
# File: linalg/backend/python_backend.py
# ==============================
"""Pure-Python backend — no external dependencies beyond the stdlib.

Every operation uses plain ``list`` arithmetic so that the engine is
fully self-contained when numpy is not available or when the user
explicitly requests ``"python"`` mode.
"""

from __future__ import annotations

import math

from mllense.math.linalg.backend.base import Backend
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.exceptions import (
    EmptyMatrixError,
    ShapeMismatchError,
    SingularMatrixError,
)
from mllense.math.linalg._internal.constants import SINGULAR_PIVOT_THRESHOLD

__all__ = ["PythonBackend"]


class PythonBackend(Backend):
    """Pure-Python reference backend."""

    @property
    def name(self) -> str:
        return "python"

    # ── matmul ────────────────────────────────────────────────────────── #

    def matmul(self, a: InternalMatrix, b: InternalMatrix) -> InternalMatrix:
        a_rows = len(a)
        a_cols = len(a[0]) if a_rows else 0
        b_rows = len(b)
        b_cols = len(b[0]) if b_rows else 0

        if a_cols != b_rows:
            raise ShapeMismatchError(
                expected=f"A.cols ({a_cols}) == B.rows",
                got=f"B.rows = {b_rows}",
                operation="matmul",
            )

        result: InternalMatrix = [[0.0] * b_cols for _ in range(a_rows)]
        for i in range(a_rows):
            for k in range(a_cols):
                a_ik = a[i][k]
                for j in range(b_cols):
                    result[i][j] += a_ik * b[k][j]
        return result

    # ── dot ───────────────────────────────────────────────────────────── #

    def dot(self, a: InternalVector, b: InternalVector) -> float:
        if len(a) != len(b):
            raise ShapeMismatchError(
                expected=f"len(a) == len(b)",
                got=f"{len(a)} vs {len(b)}",
                operation="dot",
            )
        return math.fsum(x * y for x, y in zip(a, b))

    # ── solve (Gaussian elimination with partial pivoting) ───────────── #

    def solve(self, a: InternalMatrix, b: InternalVector) -> InternalVector:
        n = len(a)
        if n == 0:
            raise EmptyMatrixError("Cannot solve an empty system.")

        # build augmented matrix (deep copy)
        aug: list[list[float]] = [
            [a[i][j] for j in range(n)] + [b[i]] for i in range(n)
        ]

        # forward elimination with partial pivoting
        for col in range(n):
            # find pivot
            max_val = abs(aug[col][col])
            max_row = col
            for row in range(col + 1, n):
                if abs(aug[row][col]) > max_val:
                    max_val = abs(aug[row][col])
                    max_row = row
            if max_val < SINGULAR_PIVOT_THRESHOLD:
                raise SingularMatrixError(
                    f"Zero pivot encountered at column {col}. Matrix is singular."
                )
            if max_row != col:
                aug[col], aug[max_row] = aug[max_row], aug[col]

            pivot = aug[col][col]
            for row in range(col + 1, n):
                factor = aug[row][col] / pivot
                for j in range(col, n + 1):
                    aug[row][j] -= factor * aug[col][j]

        # back substitution
        x: InternalVector = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = math.fsum(aug[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (aug[i][n] - s) / aug[i][i]
        return x

    # ── inverse (via row reduction) ──────────────────────────────────── #

    def inverse(self, a: InternalMatrix) -> InternalMatrix:
        n = len(a)
        if n == 0:
            raise EmptyMatrixError("Cannot invert an empty matrix.")

        # augment with identity
        aug: list[list[float]] = [
            [a[i][j] for j in range(n)] + [1.0 if i == k else 0.0 for k in range(n)]
            for i in range(n)
        ]

        for col in range(n):
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

        return [[aug[i][j + n] for j in range(n)] for i in range(n)]

    # ── transpose ─────────────────────────────────────────────────────── #

    def transpose(self, a: InternalMatrix) -> InternalMatrix:
        rows = len(a)
        if rows == 0:
            return []
        cols = len(a[0])
        return [[a[i][j] for i in range(rows)] for j in range(cols)]

    # ── factory helpers ──────────────────────────────────────────────── #

    def zeros(self, rows: int, cols: int) -> InternalMatrix:
        return [[0.0] * cols for _ in range(rows)]

    def identity(self, n: int) -> InternalMatrix:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
