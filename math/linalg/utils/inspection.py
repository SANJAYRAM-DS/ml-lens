# ==============================
# File: linalg/utils/inspection.py
# ==============================
"""Matrix introspection and property-checking utilities."""

from __future__ import annotations

from mllense.math.linalg._internal.constants import DEFAULT_FLOAT_TOLERANCE
from mllense.math.linalg.core.types import InternalMatrix, get_matrix_shape

__all__ = [
    "describe_matrix",
    "is_symmetric",
    "is_upper_triangular",
    "is_lower_triangular",
    "is_diagonal",
    "is_identity",
]


def describe_matrix(m: InternalMatrix) -> dict[str, object]:
    """Return a dict of human-readable matrix properties."""
    rows, cols = get_matrix_shape(m)
    return {
        "rows": rows,
        "cols": cols,
        "square": rows == cols,
        "symmetric": is_symmetric(m) if rows == cols else False,
        "diagonal": is_diagonal(m) if rows == cols else False,
        "identity": is_identity(m) if rows == cols else False,
        "upper_triangular": is_upper_triangular(m) if rows == cols else False,
        "lower_triangular": is_lower_triangular(m) if rows == cols else False,
    }


def is_symmetric(m: InternalMatrix, tol: float = DEFAULT_FLOAT_TOLERANCE) -> bool:
    """Check if a square matrix is symmetric."""
    n = len(m)
    if n == 0:
        return True
    if len(m[0]) != n:
        return False
    for i in range(n):
        for j in range(i + 1, n):
            if abs(m[i][j] - m[j][i]) > tol:
                return False
    return True


def is_upper_triangular(m: InternalMatrix, tol: float = DEFAULT_FLOAT_TOLERANCE) -> bool:
    """Check if a square matrix is upper-triangular."""
    n = len(m)
    for i in range(1, n):
        for j in range(0, i):
            if abs(m[i][j]) > tol:
                return False
    return True


def is_lower_triangular(m: InternalMatrix, tol: float = DEFAULT_FLOAT_TOLERANCE) -> bool:
    """Check if a square matrix is lower-triangular."""
    n = len(m)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(m[i][j]) > tol:
                return False
    return True


def is_diagonal(m: InternalMatrix, tol: float = DEFAULT_FLOAT_TOLERANCE) -> bool:
    """Check if a square matrix is diagonal."""
    return is_upper_triangular(m, tol) and is_lower_triangular(m, tol)


def is_identity(m: InternalMatrix, tol: float = DEFAULT_FLOAT_TOLERANCE) -> bool:
    """Check if a square matrix is the identity matrix."""
    n = len(m)
    if n == 0:
        return True
    for i in range(n):
        for j in range(n):
            expected = 1.0 if i == j else 0.0
            if abs(m[i][j] - expected) > tol:
                return False
    return True
