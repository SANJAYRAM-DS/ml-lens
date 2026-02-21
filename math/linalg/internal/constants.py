# ==============================
# File: linalg/internal/constants.py
# ==============================
"""Re-export of _internal constants for backward-compatibility.

Prefer importing from ``linalg._internal.constants`` in new code.
"""

from mllense.math.linalg._internal.constants import (
    DEFAULT_FLOAT_TOLERANCE,
    FLOAT_OVERFLOW_GUARD,
    MAX_MATRIX_DIM,
    MEDIUM_MATRIX_THRESHOLD,
    RELATIVE_TOLERANCE,
    SINGULAR_PIVOT_THRESHOLD,
    SMALL_MATRIX_THRESHOLD,
)

__all__ = [
    "DEFAULT_FLOAT_TOLERANCE",
    "RELATIVE_TOLERANCE",
    "SINGULAR_PIVOT_THRESHOLD",
    "SMALL_MATRIX_THRESHOLD",
    "MEDIUM_MATRIX_THRESHOLD",
    "FLOAT_OVERFLOW_GUARD",
    "MAX_MATRIX_DIM",
]
