# ==============================
# File: linalg/diagnostics/error_analysis.py
# ==============================
"""Forward and backward error analysis for linear system solutions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from mllense.math.linalg.core.types import (
    InternalMatrix,
    InternalVector,
    MatrixLike,
    VectorLike,
    to_internal_matrix,
    to_internal_vector,
)

__all__ = ["forward_error", "backward_error"]


def forward_error(
    x_computed: VectorLike,
    x_exact: VectorLike,
) -> float:
    """Compute the relative forward error: ``||x_computed - x_exact|| / ||x_exact||``.

    Uses the 2-norm.
    """
    xc = to_internal_vector(x_computed)
    xe = to_internal_vector(x_exact)

    diff_norm = math.sqrt(math.fsum((xc[i] - xe[i]) ** 2 for i in range(len(xc))))
    exact_norm = math.sqrt(math.fsum(x ** 2 for x in xe))

    if exact_norm < 1e-15:
        return float("inf") if diff_norm > 1e-15 else 0.0

    return diff_norm / exact_norm


def backward_error(
    a: MatrixLike,
    b: VectorLike,
    x_computed: VectorLike,
) -> float:
    """Compute the relative backward error: ``||b - Ax|| / ||b||``.

    Measures how well the computed solution satisfies the original system.
    """
    a_int = to_internal_matrix(a)
    b_int = to_internal_vector(b)
    x_int = to_internal_vector(x_computed)

    n = len(a_int)
    # compute residual r = b - Ax
    residual: list[float] = []
    for i in range(n):
        ax_i = math.fsum(a_int[i][j] * x_int[j] for j in range(len(x_int)))
        residual.append(b_int[i] - ax_i)

    r_norm = math.sqrt(math.fsum(r ** 2 for r in residual))
    b_norm = math.sqrt(math.fsum(x ** 2 for x in b_int))

    if b_norm < 1e-15:
        return float("inf") if r_norm > 1e-15 else 0.0

    return r_norm / b_norm
