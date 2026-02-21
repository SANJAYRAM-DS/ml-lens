# ==============================
# File: linalg/diagnostics/rank.py
# ==============================
"""Matrix rank estimation via SVD."""

from __future__ import annotations

from typing import Optional

import numpy as np

from mllense.math.linalg.core.types import MatrixLike, to_internal_matrix

__all__ = ["matrix_rank"]


def matrix_rank(
    a: MatrixLike,
    *,
    tol: Optional[float] = None,
) -> int:
    """Estimate the rank of a matrix using SVD.

    Args:
        a: Input matrix.
        tol: Tolerance below which singular values are treated as zero.
             Defaults to ``max(m, n) * eps * σ_max``.

    Returns:
        The estimated rank.
    """
    a_int = to_internal_matrix(a)
    a_np = np.array(a_int, dtype=np.float64)

    s = np.linalg.svd(a_np, compute_uv=False)

    if tol is None:
        m, n = a_np.shape
        eps = np.finfo(np.float64).eps
        tol = max(m, n) * eps * (s[0] if len(s) > 0 else 0.0)

    return int(np.sum(s > tol))
