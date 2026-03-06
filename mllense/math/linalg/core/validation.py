# ==============================
# File: linalg/core/validation.py
# ==============================
"""Shape and value validation utilities.

Every validation function raises a domain-specific exception from
``linalg.exceptions`` — never a raw ``ValueError``.
"""

from __future__ import annotations

from typing import Sequence

from mllense.math.linalg._internal.constants import MAX_MATRIX_DIM
from mllense.math.linalg.core.types import InternalMatrix, InternalVector, get_matrix_shape
from mllense.math.linalg.exceptions import (
    EmptyMatrixError,
    InvalidInputError,
    ShapeMismatchError,
)

__all__ = [
    "validate_matmul_shapes",
    "validate_square",
    "validate_solve_shapes",
    "validate_dimension_limit",
]


def validate_dimension_limit(rows: int, cols: int) -> None:
    """Raise if dimensions exceed the safety limit."""
    if rows > MAX_MATRIX_DIM or cols > MAX_MATRIX_DIM:
        raise InvalidInputError(
            f"Matrix dimension ({rows}×{cols}) exceeds maximum "
            f"allowed dimension {MAX_MATRIX_DIM}."
        )


def validate_matmul_shapes(
    a_shape: tuple[int, int],
    b_shape: tuple[int, int],
) -> tuple[int, int, int]:
    """Validate shapes for matrix multiplication ``A @ B``.

    Returns ``(m, k, n)`` where the result is ``m × n`` and the inner
    dimension is ``k``.

    Raises:
        ShapeMismatchError: When ``a_cols != b_rows``.
        EmptyMatrixError: When any dimension is zero.
    """
    a_rows, a_cols = a_shape
    b_rows, b_cols = b_shape

    if a_rows == 0 or a_cols == 0 or b_rows == 0 or b_cols == 0:
        raise EmptyMatrixError("Cannot multiply matrices with zero-sized dimensions.")

    if a_cols != b_rows:
        raise ShapeMismatchError(
            expected=f"A.cols == B.rows (inner dimension)",
            got=f"A is {a_rows}×{a_cols}, B is {b_rows}×{b_cols}",
            operation="matmul",
        )
    return a_rows, a_cols, b_cols


def validate_square(m: InternalMatrix, operation: str = "") -> int:
    """Validate that the matrix is square. Return the dimension *n*."""
    rows, cols = get_matrix_shape(m)
    if rows == 0 or cols == 0:
        raise EmptyMatrixError("Cannot operate on an empty matrix.")
    if rows != cols:
        raise ShapeMismatchError(
            expected="square matrix (rows == cols)",
            got=f"{rows}×{cols}",
            operation=operation,
        )
    return rows


def validate_solve_shapes(
    a: InternalMatrix, b: InternalVector | InternalMatrix
) -> int:
    """Validate shapes for ``solve(A, b)``.

    ``A`` must be square ``n × n`` and ``b`` must have length ``n``
    (or be an ``n × k`` matrix for multiple RHS).

    Returns:
        The dimension *n*.
    """
    n = validate_square(a, operation="solve")

    # b can be a vector (list[float]) or matrix (list[list[float]])
    if isinstance(b, list) and len(b) > 0 and isinstance(b[0], list):
        # matrix RHS
        b_rows = len(b)
    else:
        b_rows = len(b)  # type: ignore[arg-type]

    if b_rows != n:
        raise ShapeMismatchError(
            expected=f"b length == {n}",
            got=f"b length == {b_rows}",
            operation="solve",
        )
    return n
