# ==============================
# File: linalg/diagnostics/condition_number.py
# ==============================
"""Condition number computation."""

from __future__ import annotations

from typing import Optional

import numpy as np

from mllense.math.linalg.core.types import InternalMatrix, MatrixLike, to_internal_matrix

__all__ = ["condition_number"]


def condition_number(
    a: MatrixLike,
    *,
    ord: Optional[int] = None,
) -> float:
    """Compute the condition number of a matrix.

    Uses SVD: ``cond(A) = σ_max / σ_min``.

    Args:
        a: Input matrix.
        ord: Norm order (default: 2-norm via SVD).

    Returns:
        The condition number.  Returns ``float('inf')`` for singular matrices.
    """
    a_int = to_internal_matrix(a)
    a_np = np.array(a_int, dtype=np.float64)

    if ord is not None:
        return float(np.linalg.cond(a_np, p=ord))

    s = np.linalg.svd(a_np, compute_uv=False)
    if len(s) == 0 or s[-1] < 1e-15:
        return float("inf")
    return float(s[0] / s[-1])
