# ==============================
# File: linalg/algorithms/shape/reshape.py
# ==============================
"""Reshape a matrix to a new shape (row-major)."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.shape.base import BaseShape
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import ShapeMismatchError

__all__ = ["Reshape"]


class Reshape(BaseShape):
    """Reshape matrix to (new_rows, new_cols) in row-major order."""

    metadata = AlgorithmMetadata(
        name="reshape",
        operation="reshape",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Reshape a matrix to new dimensions (row-major).",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        m: InternalMatrix = args[0]
        new_rows: int = args[1]
        new_cols: int = args[2]

        rows = len(m)
        cols = len(m[0]) if rows else 0
        total = rows * cols

        if new_rows * new_cols != total:
            raise ShapeMismatchError(
                expected=f"product = {total}",
                got=f"{new_rows}×{new_cols} = {new_rows * new_cols}",
                operation="reshape",
            )

        trace.record(
            operation="reshape",
            description=f"Reshaping {rows}×{cols} → {new_rows}×{new_cols}",
        )

        flat = [v for row in m for v in row]
        return [flat[i * new_cols: (i + 1) * new_cols] for i in range(new_rows)]
