# ==============================
# File: linalg/diagnostics/__init__.py
# ==============================
"""Diagnostics subsystem — condition number, rank, stability analysis."""

from mllense.math.linalg.diagnostics.condition_number import condition_number
from mllense.math.linalg.diagnostics.rank import matrix_rank
from mllense.math.linalg.diagnostics.stability import stability_report
from mllense.math.linalg.diagnostics.error_analysis import forward_error, backward_error
from mllense.math.linalg.diagnostics.report import full_diagnostic_report

__all__ = [
    "condition_number",
    "matrix_rank",
    "stability_report",
    "forward_error",
    "backward_error",
    "full_diagnostic_report",
]
