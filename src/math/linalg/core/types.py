# ==============================
# File: linalg/core/types.py
# ==============================
"""Canonical type aliases and input-normalisation helpers.

Every algorithm / backend works with ``list[list[float]]`` internally.
Conversion from and to numpy arrays happens at the boundary (API layer).
"""

from __future__ import annotations

import math
from typing import Any, List, Sequence, Union

import numpy as np

__all__ = [
    "MatrixLike",
    "VectorLike",
    "Scalar",
    "InternalMatrix",
    "InternalVector",
    "to_internal_matrix",
    "to_internal_vector",
    "from_internal_matrix",
    "from_internal_vector",
    "is_numpy",
    "get_matrix_shape",
    "get_vector_length",
]

# ── Public type aliases ─────────────────────────────────────────────────── #
Scalar = Union[int, float, np.integer, np.floating]
VectorLike = Union[List[float], np.ndarray, Sequence[float]]
MatrixLike = Union[List[List[float]], np.ndarray, Sequence[Sequence[float]]]

# ── Internal canonical forms ────────────────────────────────────────────── #
InternalMatrix = List[List[float]]
InternalVector = List[float]


# ── Helpers ──────────────────────────────────────────────────────────────── #

def is_numpy(obj: Any) -> bool:
    """Return ``True`` if *obj* is an ``ndarray``."""
    return isinstance(obj, np.ndarray)


def _assert_numeric(value: Any, label: str = "element") -> float:
    """Convert a single value to ``float``, raising on non-numeric."""
    from mllense.math.linalg.exceptions import InvalidInputError

    if isinstance(value, (int, float, np.integer, np.floating)):
        f = float(value)
        if math.isnan(f):
            raise InvalidInputError(f"{label} is NaN.")
        return f
    raise InvalidInputError(
        f"Non-numeric {label}: {value!r} (type {type(value).__name__})"
    )


def to_internal_vector(v: VectorLike) -> InternalVector:
    """Normalise a 1-D input to ``list[float]``."""
    from mllense.math.linalg.exceptions import EmptyMatrixError, InvalidInputError

    if isinstance(v, np.ndarray):
        if v.ndim != 1:
            raise InvalidInputError(
                f"Expected 1-D array for vector, got {v.ndim}-D."
            )
        if v.size == 0:
            raise EmptyMatrixError("Empty vector is not supported.")
        return [_assert_numeric(x, f"vector[{i}]") for i, x in enumerate(v)]

    if not isinstance(v, (list, tuple)):
        raise InvalidInputError(
            f"Expected list or ndarray for vector, got {type(v).__name__}."
        )
    if len(v) == 0:
        raise EmptyMatrixError("Empty vector is not supported.")
    return [_assert_numeric(x, f"vector[{i}]") for i, x in enumerate(v)]


def to_internal_matrix(m: MatrixLike) -> InternalMatrix:
    """Normalise a 2-D input to ``list[list[float]]``."""
    from mllense.math.linalg.exceptions import (
        EmptyMatrixError,
        InvalidInputError,
        NonRectangularMatrixError,
    )

    if isinstance(m, np.ndarray):
        if m.ndim == 1:
            # treat 1-D array as a row vector → shape (1, n)
            if m.size == 0:
                raise EmptyMatrixError("Empty matrix is not supported.")
            return [[_assert_numeric(x, f"matrix[0][{j}]") for j, x in enumerate(m)]]
        if m.ndim != 2:
            raise InvalidInputError(
                f"Expected 2-D array for matrix, got {m.ndim}-D."
            )
        if m.shape[0] == 0 or m.shape[1] == 0:
            raise EmptyMatrixError("Empty matrix is not supported.")
        return [
            [_assert_numeric(m[i, j], f"matrix[{i}][{j}]") for j in range(m.shape[1])]
            for i in range(m.shape[0])
        ]

    if not isinstance(m, (list, tuple)):
        raise InvalidInputError(
            f"Expected list-of-lists or ndarray for matrix, got {type(m).__name__}."
        )
    if len(m) == 0:
        raise EmptyMatrixError("Empty matrix is not supported.")

    # validate rectangularity
    row_lengths: list[int] = []
    for i, row in enumerate(m):
        if not isinstance(row, (list, tuple, np.ndarray)):
            raise InvalidInputError(
                f"Row {i} is not a list/tuple/ndarray: {type(row).__name__}."
            )
        row_lengths.append(len(row))

    if len(set(row_lengths)) != 1:
        raise NonRectangularMatrixError(row_lengths)

    if row_lengths[0] == 0:
        raise EmptyMatrixError("Empty matrix is not supported (zero-width rows).")

    return [
        [_assert_numeric(m[i][j], f"matrix[{i}][{j}]") for j in range(row_lengths[0])]
        for i in range(len(m))
    ]


def from_internal_matrix(
    internal: InternalMatrix, *, as_numpy: bool = False
) -> MatrixLike:
    """Convert an internal matrix back to the caller's preferred format."""
    if as_numpy:
        return np.array(internal, dtype=np.float64)
    return internal


def from_internal_vector(
    internal: InternalVector, *, as_numpy: bool = False
) -> VectorLike:
    """Convert an internal vector back to the caller's preferred format."""
    if as_numpy:
        return np.array(internal, dtype=np.float64)
    return internal


def get_matrix_shape(m: InternalMatrix) -> tuple[int, int]:
    """Return ``(rows, cols)`` of an already-validated internal matrix."""
    rows = len(m)
    cols = len(m[0]) if rows > 0 else 0
    return rows, cols


def get_vector_length(v: InternalVector) -> int:
    """Return the length of an already-validated internal vector."""
    return len(v)
