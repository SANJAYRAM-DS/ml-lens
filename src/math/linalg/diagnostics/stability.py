# ==============================
# File: linalg/diagnostics/stability.py
# ==============================
"""Stability analysis for linear systems."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from mllense.math.linalg.core.types import MatrixLike, to_internal_matrix
from mllense.math.linalg.diagnostics.condition_number import condition_number
from mllense.math.linalg.diagnostics.rank import matrix_rank

__all__ = ["stability_report"]


def stability_report(a: MatrixLike) -> Dict[str, Any]:
    """Generate a stability report for a matrix.

    Reports:
    - Condition number (2-norm)
    - Rank
    - Whether the matrix is well-conditioned
    - Estimated digits of accuracy lost
    """
    a_int = to_internal_matrix(a)
    a_np = np.array(a_int, dtype=np.float64)

    cond = condition_number(a_int)
    rnk = matrix_rank(a_int)
    rows, cols = a_np.shape

    import math
    if cond == float("inf"):
        digits_lost = float("inf")
        well_conditioned = False
    else:
        digits_lost = math.log10(cond) if cond > 0 else 0.0
        well_conditioned = cond < 1e6

    return {
        "shape": (rows, cols),
        "rank": rnk,
        "full_rank": rnk == min(rows, cols),
        "condition_number": cond,
        "well_conditioned": well_conditioned,
        "estimated_digits_lost": round(digits_lost, 2),
        "recommendation": (
            "Matrix is well-conditioned — results should be reliable."
            if well_conditioned
            else "Matrix is ill-conditioned — results may have significant error."
        ),
    }
