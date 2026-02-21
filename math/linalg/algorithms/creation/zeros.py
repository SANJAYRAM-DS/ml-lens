# ==============================
# File: linalg/algorithms/creation/zeros.py
# ==============================
"""Create a matrix of zeros."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.creation.base import BaseCreation
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import InvalidInputError

__all__ = ["ZerosCreation"]


class ZerosCreation(BaseCreation):
    """Create an ``m × n`` zero matrix."""

    metadata = AlgorithmMetadata(
        name="zeros",
        operation="create_zeros",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Creates a matrix filled with zeros.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        """Create zero matrix.

        Args:
            args[0]: rows (int)
            args[1]: cols (int)
        """
        rows: int = args[0]
        cols: int = args[1] if len(args) > 1 else rows

        if rows <= 0 or cols <= 0:
            raise InvalidInputError(
                f"Matrix dimensions must be positive, got ({rows}, {cols})."
            )

        trace.record(
            operation="zeros",
            description=f"Creating {rows}×{cols} zero matrix",
        )

        return [[0.0] * cols for _ in range(rows)]
