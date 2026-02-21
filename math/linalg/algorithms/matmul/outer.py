# ==============================
# File: linalg/algorithms/matmul/outer.py
# ==============================
"""Outer product of two vectors."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.matmul.base import BaseMatmul
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector

__all__ = ["OuterProduct"]


class OuterProduct(BaseMatmul):
    """Outer product: ``C[i][j] = a[i] * b[j]``."""

    metadata = AlgorithmMetadata(
        name="outer",
        operation="outer",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Vector outer product producing an m×n matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        a: InternalVector = args[0]
        b: InternalVector = args[1]

        m = len(a)
        n = len(b)

        trace.record(
            operation="outer_start",
            description=f"Outer product: ({m},) ⊗ ({n},) → {m}×{n}",
        )

        result: InternalMatrix = [[a[i] * b[j] for j in range(n)] for i in range(m)]

        trace.record(
            operation="outer_done",
            description=f"Result shape: {m}×{n}",
            data=result,
        )

        return result
