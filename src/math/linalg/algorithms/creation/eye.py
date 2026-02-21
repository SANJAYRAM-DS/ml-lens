# ==============================
# File: linalg/algorithms/creation/eye.py
# ==============================
"""Create an identity matrix."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.creation.base import BaseCreation
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import InvalidInputError

__all__ = ["EyeCreation"]


class EyeCreation(BaseCreation):
    """Create an ``n × m`` identity-like matrix (ones on the main diagonal)."""

    metadata = AlgorithmMetadata(
        name="eye",
        operation="create_eye",
        complexity="O(n*m)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Creates an identity (or rectangular identity-like) matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        """Create identity matrix.

        Args:
            args[0]: rows (int)
            args[1]: cols (int, defaults to rows)
        """
        rows: int = args[0]
        cols: int = args[1] if len(args) > 1 else rows

        if rows <= 0 or cols <= 0:
            raise InvalidInputError(
                f"Matrix dimensions must be positive, got ({rows}, {cols})."
            )

        trace.record(
            operation="eye",
            description=f"Creating {rows}×{cols} identity matrix",
        )

        return [
            [1.0 if i == j else 0.0 for j in range(cols)]
            for i in range(rows)
        ]
