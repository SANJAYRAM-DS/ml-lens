# ==============================
# File: linalg/algorithms/creation/rand.py
# ==============================
"""Create a matrix with random values."""

from __future__ import annotations

import random as _random
from typing import Any, Optional

from mllense.math.linalg.algorithms.creation.base import BaseCreation
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import InvalidInputError

__all__ = ["RandCreation"]


class RandCreation(BaseCreation):
    """Create an ``m × n`` matrix with uniform-random values in ``[0, 1)``."""

    metadata = AlgorithmMetadata(
        name="rand",
        operation="create_rand",
        complexity="O(m*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Creates a matrix filled with uniform random values in [0, 1).",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        """Create random matrix.

        Args:
            args[0]: rows (int)
            args[1]: cols (int, defaults to rows)

        Keyword Args:
            seed: Optional random seed for reproducibility.
            low: Lower bound (default 0.0).
            high: Upper bound (default 1.0).
        """
        rows: int = args[0]
        cols: int = args[1] if len(args) > 1 else rows
        seed: Optional[int] = kwargs.get("seed", None)
        low: float = kwargs.get("low", 0.0)
        high: float = kwargs.get("high", 1.0)

        if rows <= 0 or cols <= 0:
            raise InvalidInputError(
                f"Matrix dimensions must be positive, got ({rows}, {cols})."
            )

        if seed is not None:
            _random.seed(seed)

        trace.record(
            operation="rand",
            description=f"Creating {rows}×{cols} random matrix (range [{low}, {high}))",
        )

        span = high - low
        return [
            [low + _random.random() * span for _ in range(cols)]
            for _ in range(rows)
        ]
