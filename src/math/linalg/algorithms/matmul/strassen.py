# ==============================
# File: linalg/algorithms/matmul/strassen.py
# ==============================
"""Strassen's algorithm for matrix multiplication.

Complexity: O(n^2.807) — sub-cubic but with higher constant factors
and memory overhead.  Falls back to naive for small sub-problems.
"""

from __future__ import annotations

from typing import Any, Union

from mllense.math.linalg.algorithms.matmul.base import BaseMatmul
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_matmul_shapes

__all__ = ["StrassenMatmul"]

_STRASSEN_THRESHOLD = 64  # fall back to naive below this


class StrassenMatmul(BaseMatmul):
    """Strassen's recursive matrix multiplication."""

    metadata = AlgorithmMetadata(
        name="strassen_matmul",
        operation="matmul",
        complexity="O(n^2.807)",
        stable=False,
        supports_batch=False,
        requires_square=False,
        description=(
            "Strassen's algorithm — sub-cubic O(n^2.807) but with higher "
            "constants and numerical instability for large problems."
        ),
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Union[InternalMatrix, InternalVector, float]:
        a: InternalMatrix = args[0]
        b: InternalMatrix = args[1]

        a_rows = len(a)
        a_cols = len(a[0]) if a_rows else 0
        b_rows = len(b)
        b_cols = len(b[0]) if b_rows else 0

        m, k, n = validate_matmul_shapes((a_rows, a_cols), (b_rows, b_cols))

        trace.record(
            operation="strassen_start",
            description=f"Strassen matmul: ({m}×{k}) @ ({k}×{n})",
        )

        # Pad to power of 2
        max_dim = max(m, k, n)
        size = 1
        while size < max_dim:
            size *= 2

        a_pad = _pad(a, size, size)
        b_pad = _pad(b, size, size)

        result_pad = _strassen_recursive(a_pad, b_pad, size)

        # Unpad
        result = [row[:n] for row in result_pad[:m]]

        trace.record(
            operation="strassen_done",
            description=f"Result shape: ({m}×{n})",
        )

        if m == 1 and n == 1:
            return result[0][0]
        if n == 1:
            return [row[0] for row in result]
        if m == 1:
            return result[0]
        return result


def _pad(m: InternalMatrix, rows: int, cols: int) -> InternalMatrix:
    """Pad matrix with zeros to size (rows × cols)."""
    orig_rows = len(m)
    orig_cols = len(m[0]) if orig_rows else 0
    result: InternalMatrix = []
    for i in range(rows):
        if i < orig_rows:
            row = m[i][:] + [0.0] * (cols - orig_cols)
        else:
            row = [0.0] * cols
        result.append(row)
    return result


def _mat_add(a: InternalMatrix, b: InternalMatrix, n: int) -> InternalMatrix:
    return [[a[i][j] + b[i][j] for j in range(n)] for i in range(n)]


def _mat_sub(a: InternalMatrix, b: InternalMatrix, n: int) -> InternalMatrix:
    return [[a[i][j] - b[i][j] for j in range(n)] for i in range(n)]


def _naive_mul(a: InternalMatrix, b: InternalMatrix, n: int) -> InternalMatrix:
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            a_ik = a[i][k]
            if a_ik == 0.0:
                continue
            for j in range(n):
                result[i][j] += a_ik * b[k][j]
    return result


def _split(m: InternalMatrix, n: int) -> tuple[
    InternalMatrix, InternalMatrix, InternalMatrix, InternalMatrix
]:
    """Split an n×n matrix into four (n/2)×(n/2) quadrants."""
    h = n // 2
    a11 = [row[:h] for row in m[:h]]
    a12 = [row[h:] for row in m[:h]]
    a21 = [row[:h] for row in m[h:]]
    a22 = [row[h:] for row in m[h:]]
    return a11, a12, a21, a22


def _merge(c11: InternalMatrix, c12: InternalMatrix,
           c21: InternalMatrix, c22: InternalMatrix, h: int) -> InternalMatrix:
    """Merge four quadrants back into one matrix."""
    top = [c11[i] + c12[i] for i in range(h)]
    bottom = [c21[i] + c22[i] for i in range(h)]
    return top + bottom


def _strassen_recursive(a: InternalMatrix, b: InternalMatrix, n: int) -> InternalMatrix:
    """Recursive Strassen multiplication for n×n matrices (n is a power of 2)."""
    if n <= _STRASSEN_THRESHOLD:
        return _naive_mul(a, b, n)

    h = n // 2

    a11, a12, a21, a22 = _split(a, n)
    b11, b12, b21, b22 = _split(b, n)

    m1 = _strassen_recursive(_mat_add(a11, a22, h), _mat_add(b11, b22, h), h)
    m2 = _strassen_recursive(_mat_add(a21, a22, h), b11, h)
    m3 = _strassen_recursive(a11, _mat_sub(b12, b22, h), h)
    m4 = _strassen_recursive(a22, _mat_sub(b21, b11, h), h)
    m5 = _strassen_recursive(_mat_add(a11, a12, h), b22, h)
    m6 = _strassen_recursive(_mat_sub(a21, a11, h), _mat_add(b11, b12, h), h)
    m7 = _strassen_recursive(_mat_sub(a12, a22, h), _mat_add(b21, b22, h), h)

    c11 = _mat_add(_mat_sub(_mat_add(m1, m4, h), m5, h), m7, h)
    c12 = _mat_add(m3, m5, h)
    c21 = _mat_add(m2, m4, h)
    c22 = _mat_add(_mat_sub(_mat_add(m1, m3, h), m2, h), m6, h)

    return _merge(c11, c12, c21, c22, h)
