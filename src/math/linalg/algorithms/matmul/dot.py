# ==============================
# File: linalg/algorithms/matmul/dot.py
# ==============================
"""Dot (inner) product of two vectors."""

from __future__ import annotations

import math
from typing import Any, Union

from mllense.math.linalg.algorithms.matmul.base import BaseMatmul
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalVector
from mllense.math.linalg.exceptions import ShapeMismatchError

__all__ = ["DotProduct"]


class DotProduct(BaseMatmul):
    """Inner (dot) product: ``result = Σ_i a[i] * b[i]``."""

    metadata = AlgorithmMetadata(
        name="dot",
        operation="dot",
        complexity="O(n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Vector dot (inner) product using compensated (Kahan) summation.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> float:
        a: InternalVector = args[0]
        b: InternalVector = args[1]

        if len(a) != len(b):
            raise ShapeMismatchError(
                expected=f"same length ({len(a)})",
                got=f"length {len(b)}",
                operation="dot",
            )

        trace.record(
            operation="dot_start",
            description=f"Dot product of vectors (length {len(a)})",
        )

        result = math.fsum(x * y for x, y in zip(a, b))

        trace.record(
            operation="dot_done",
            description=f"Result: {result}",
        )

        return result
