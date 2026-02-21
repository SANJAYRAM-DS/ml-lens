# ==============================
# File: linalg/diagnostics/report.py
# ==============================
"""Full diagnostic report combining all analysis tools."""

from __future__ import annotations

from typing import Any, Dict

from mllense.math.linalg.core.types import MatrixLike, to_internal_matrix
from mllense.math.linalg.diagnostics.condition_number import condition_number
from mllense.math.linalg.diagnostics.rank import matrix_rank
from mllense.math.linalg.diagnostics.stability import stability_report
from mllense.math.linalg.utils.inspection import describe_matrix

__all__ = ["full_diagnostic_report"]


def full_diagnostic_report(a: MatrixLike) -> Dict[str, Any]:
    """Generate a comprehensive diagnostic report for a matrix.

    Combines:
    - Matrix properties (shape, symmetry, etc.)
    - Stability analysis (condition number, rank)
    - Recommendations
    """
    a_int = to_internal_matrix(a)

    props = describe_matrix(a_int)
    stab = stability_report(a_int)

    return {
        "properties": props,
        "stability": stab,
        "summary": {
            "shape": f"{props['rows']}×{props['cols']}",
            "square": props["square"],
            "symmetric": props["symmetric"],
            "rank": stab["rank"],
            "full_rank": stab["full_rank"],
            "condition_number": stab["condition_number"],
            "well_conditioned": stab["well_conditioned"],
            "recommendation": stab["recommendation"],
        },
    }
