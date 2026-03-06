# ==============================
# File: linalg/algorithms/norms/frobenius.py
# ==============================
"""Frobenius norm: sqrt(sum of squared elements)."""

from __future__ import annotations

import math
from typing import Any

from mllense.math.linalg.algorithms.norms.base import BaseNorm
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix

__all__ = ["FrobeniusNorm"]


class FrobeniusNorm(BaseNorm):
    """Frobenius norm: ``||A||_F = sqrt(Σ_{i,j} a_{i,j}^2)``."""

    metadata = AlgorithmMetadata(
        name="frobenius_norm",
        operation="norm_frobenius",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Frobenius (element-wise L2) matrix norm.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> float:
        m: InternalMatrix = args[0]
        rows = len(m)
        cols = len(m[0]) if rows else 0

        trace.record(
            operation="frobenius_start",
            description=f"Computing Frobenius norm of {rows}×{cols} matrix",
        )

        result = math.sqrt(math.fsum(m[i][j] ** 2 for i in range(rows) for j in range(cols)))

        trace.record(
            operation="frobenius_done",
            description=f"Frobenius norm = {result}",
        )

        return result
