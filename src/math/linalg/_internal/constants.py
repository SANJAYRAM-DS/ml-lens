# ==============================
# File: linalg/_internal/constants.py
# ==============================
"""Core numerical and configuration constants for the linalg engine."""

import math

__all__ = [
    "DEFAULT_FLOAT_TOLERANCE",
    "RELATIVE_TOLERANCE",
    "SINGULAR_PIVOT_THRESHOLD",
    "FLOAT_OVERFLOW_GUARD",
    "MAX_MATRIX_DIM",
    "SMALL_MATRIX_THRESHOLD",
    "MEDIUM_MATRIX_THRESHOLD",
]

# ── Tolerances ──────────────────────────────────────────────────────────── #
DEFAULT_FLOAT_TOLERANCE: float = 1e-12
RELATIVE_TOLERANCE: float = 1e-9

# A pivot smaller than this is considered singular
SINGULAR_PIVOT_THRESHOLD: float = 1e-14

# Blowup guard for native Python lists before math.inf is hit
FLOAT_OVERFLOW_GUARD: float = 1e308


# ── Shape Limits ────────────────────────────────────────────────────────── #
# Prevent accidental memory exhaustion with Python lists
MAX_MATRIX_DIM: int = 10000

# Below this size, bypass cache blocking and fast paths
SMALL_MATRIX_THRESHOLD: int = 64

# Between small and medium, use blocking; beyond medium, fall back to BLAS via numpy if possible
MEDIUM_MATRIX_THRESHOLD: int = 512
