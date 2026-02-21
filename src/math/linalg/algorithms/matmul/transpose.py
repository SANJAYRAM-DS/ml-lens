# ==============================
# File: linalg/algorithms/matmul/transpose.py
# ==============================
"""Matrix transpose algorithm."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.matmul.base import BaseMatmul
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import EmptyMatrixError

__all__ = ["Transpose"]


class Transpose(BaseMatmul):
    """Matrix transpose: ``B[j][i] = A[i][j]``."""

    metadata = AlgorithmMetadata(
        name="transpose",
        operation="transpose",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Matrix transpose.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        m: InternalMatrix = args[0]

        rows = len(m)
        if rows == 0:
            raise EmptyMatrixError("Cannot transpose an empty matrix.")
        cols = len(m[0])

        trace.record(
            operation="transpose",
            description=f"Transposing {rows}×{cols} → {cols}×{rows}",
        )

        return [[m[i][j] for i in range(rows)] for j in range(cols)]
