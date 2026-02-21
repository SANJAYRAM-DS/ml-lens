# ==============================
# File: linalg/algorithms/shape/flatten.py
# ==============================
"""Flatten a matrix to a 1-D vector (row-major)."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.shape.base import BaseShape
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector

__all__ = ["Flatten"]


class Flatten(BaseShape):
    """Flatten a 2-D matrix to a 1-D vector."""

    metadata = AlgorithmMetadata(
        name="flatten",
        operation="flatten",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Flatten a matrix into a 1-D vector (row-major).",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalVector:
        m: InternalMatrix = args[0]

        rows = len(m)
        cols = len(m[0]) if rows else 0

        trace.record(
            operation="flatten",
            description=f"Flattening {rows}×{cols} → ({rows * cols},)",
        )

        return [v for row in m for v in row]
