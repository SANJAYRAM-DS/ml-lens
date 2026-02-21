# ==============================
# File: linalg/utils/matrix_helpers.py
# ==============================
"""Common matrix/vector helper operations used across the engine."""

from __future__ import annotations

from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.exceptions import ShapeMismatchError

__all__ = [
    "copy_matrix",
    "copy_vector",
    "add_matrices",
    "subtract_matrices",
    "scale_matrix",
    "flatten_matrix",
    "vector_norm",
    "scale_vector",
    "add_vectors",
    "subtract_vectors",
]


def copy_matrix(m: InternalMatrix) -> InternalMatrix:
    """Return a deep copy of a matrix."""
    return [row[:] for row in m]


def copy_vector(v: InternalVector) -> InternalVector:
    """Return a copy of a vector."""
    return v[:]


def add_matrices(a: InternalMatrix, b: InternalMatrix) -> InternalMatrix:
    """Element-wise addition of two matrices."""
    _check_same_shape(a, b, "add")
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def subtract_matrices(a: InternalMatrix, b: InternalMatrix) -> InternalMatrix:
    """Element-wise subtraction: ``A - B``."""
    _check_same_shape(a, b, "subtract")
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def scale_matrix(m: InternalMatrix, scalar: float) -> InternalMatrix:
    """Multiply every element of a matrix by a scalar."""
    return [[v * scalar for v in row] for row in m]


def flatten_matrix(m: InternalMatrix) -> InternalVector:
    """Flatten a 2-D matrix to a 1-D list (row-major)."""
    return [v for row in m for v in row]


def vector_norm(v: InternalVector) -> float:
    """Euclidean (L2) norm of a vector."""
    import math
    return math.sqrt(math.fsum(x * x for x in v))


def scale_vector(v: InternalVector, scalar: float) -> InternalVector:
    """Multiply every element of a vector by a scalar."""
    return [x * scalar for x in v]


def add_vectors(a: InternalVector, b: InternalVector) -> InternalVector:
    """Element-wise addition of two vectors."""
    if len(a) != len(b):
        raise ShapeMismatchError(
            expected=f"same length ({len(a)})",
            got=f"length {len(b)}",
            operation="vector_add",
        )
    return [a[i] + b[i] for i in range(len(a))]


def subtract_vectors(a: InternalVector, b: InternalVector) -> InternalVector:
    """Element-wise subtraction of two vectors."""
    if len(a) != len(b):
        raise ShapeMismatchError(
            expected=f"same length ({len(a)})",
            got=f"length {len(b)}",
            operation="vector_subtract",
        )
    return [a[i] - b[i] for i in range(len(a))]


# ── private ──────────────────────────────────────────────────────────────── #

def _check_same_shape(a: InternalMatrix, b: InternalMatrix, op: str) -> None:
    a_r, a_c = len(a), len(a[0]) if a else 0
    b_r, b_c = len(b), len(b[0]) if b else 0
    if a_r != b_r or a_c != b_c:
        raise ShapeMismatchError(
            expected=f"({a_r}×{a_c})",
            got=f"({b_r}×{b_c})",
            operation=op,
        )
