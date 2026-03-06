# ==============================
# File: linalg/algorithms/eigen/dominant.py
# ==============================
"""Dominant eigenvalue extraction (convenience wrapper around power iteration)."""

from __future__ import annotations

from typing import Any, Tuple

from mllense.math.linalg.algorithms.eigen.power_iteration import PowerIteration
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.algorithms.eigen.base import BaseEigen

__all__ = ["DominantEigen"]


class DominantEigen(BaseEigen):
    """Extract the largest-magnitude eigenvalue and its eigenvector."""

    metadata = AlgorithmMetadata(
        name="dominant_eigen",
        operation="dominant_eigen",
        complexity="O(n^2 * iterations)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Dominant eigenvalue via power iteration.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Tuple[float, InternalVector]:
        return PowerIteration().execute(*args, context=context, trace=trace, **kwargs)
