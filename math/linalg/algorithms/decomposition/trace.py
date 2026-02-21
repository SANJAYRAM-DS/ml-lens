# ==============================
# File: linalg/algorithms/decomposition/trace.py
# ==============================
"""Matrix trace (sum of diagonal elements)."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.decomposition.base import BaseDecomposition
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.core.validation import validate_square

__all__ = ["MatrixTrace"]


class MatrixTrace(BaseDecomposition):
    """Compute the trace (sum of diagonal) of a square matrix."""

    metadata = AlgorithmMetadata(
        name="matrix_trace",
        operation="trace",
        complexity="O(n)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Sum of diagonal elements of a square matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> float:
        m: InternalMatrix = args[0]
        n = validate_square(m, operation="trace")

        import math
        result = math.fsum(m[i][i] for i in range(n))

        trace.record(
            operation="matrix_trace",
            description=f"Trace of {n}×{n} matrix = {result}",
        )

        return result
