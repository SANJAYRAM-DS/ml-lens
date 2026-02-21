# ==============================
# File: linalg/backend/utils.py
# ==============================
"""Backend utility helpers.

Shared conversion and validation logic used by multiple backends.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.exceptions import InvalidInputError

__all__ = [
    "ensure_float_matrix",
    "ensure_float_vector",
    "deep_copy_matrix",
    "deep_copy_vector",
    "matrix_to_numpy",
    "vector_to_numpy",
]


def ensure_float_matrix(m: InternalMatrix) -> InternalMatrix:
    """Ensure every element is a Python float."""
    return [[float(v) for v in row] for row in m]


def ensure_float_vector(v: InternalVector) -> InternalVector:
    """Ensure every element is a Python float."""
    return [float(x) for x in v]


def deep_copy_matrix(m: InternalMatrix) -> InternalMatrix:
    """Return a deep copy of a 2-D list matrix."""
    return [row[:] for row in m]


def deep_copy_vector(v: InternalVector) -> InternalVector:
    """Return a copy of a 1-D list vector."""
    return v[:]


def matrix_to_numpy(m: InternalMatrix) -> np.ndarray:
    """Convert an internal matrix to a numpy float64 array."""
    return np.array(m, dtype=np.float64)


def vector_to_numpy(v: InternalVector) -> np.ndarray:
    """Convert an internal vector to a numpy float64 array."""
    return np.array(v, dtype=np.float64)


def validate_same_shape(
    a: InternalMatrix, b: InternalMatrix, operation: str = ""
) -> tuple[int, int]:
    """Validate that two matrices have the same shape. Return ``(rows, cols)``."""
    from mllense.math.linalg.exceptions import ShapeMismatchError

    a_rows = len(a)
    a_cols = len(a[0]) if a_rows else 0
    b_rows = len(b)
    b_cols = len(b[0]) if b_rows else 0
    if a_rows != b_rows or a_cols != b_cols:
        raise ShapeMismatchError(
            expected=f"same shape ({a_rows}×{a_cols})",
            got=f"({b_rows}×{b_cols})",
            operation=operation,
        )
    return a_rows, a_cols
