# ==============================
# File: linalg/algorithms/shape/stack.py
# ==============================
"""Stack vectors into a matrix."""

from __future__ import annotations

from typing import Any, List

from mllense.math.linalg.algorithms.shape.base import BaseShape
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.exceptions import ShapeMismatchError, InvalidInputError

__all__ = ["StackRows", "StackColumns"]


class StackRows(BaseShape):
    """Stack row-vectors into a matrix (each vector becomes a row)."""

    metadata = AlgorithmMetadata(
        name="stack_rows",
        operation="stack_rows",
        complexity="O(n*m)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Stack row-vectors into a matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        vectors: List[InternalVector] = list(args)
        if not vectors:
            raise InvalidInputError("No vectors provided.")

        length = len(vectors[0])
        for idx, v in enumerate(vectors):
            if len(v) != length:
                raise ShapeMismatchError(
                    expected=f"length {length}",
                    got=f"length {len(v)} (vector {idx})",
                    operation="stack_rows",
                )

        trace.record(
            operation="stack_rows",
            description=f"Stacking {len(vectors)} vectors (length {length}) as rows",
        )

        return [v[:] for v in vectors]


class StackColumns(BaseShape):
    """Stack column-vectors into a matrix (each vector becomes a column)."""

    metadata = AlgorithmMetadata(
        name="stack_columns",
        operation="stack_columns",
        complexity="O(n*m)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Stack column-vectors into a matrix.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        vectors: List[InternalVector] = list(args)
        if not vectors:
            raise InvalidInputError("No vectors provided.")

        length = len(vectors[0])
        for idx, v in enumerate(vectors):
            if len(v) != length:
                raise ShapeMismatchError(
                    expected=f"length {length}",
                    got=f"length {len(v)} (vector {idx})",
                    operation="stack_columns",
                )

        trace.record(
            operation="stack_columns",
            description=f"Stacking {len(vectors)} vectors (length {length}) as columns",
        )

        return [[vectors[j][i] for j in range(len(vectors))] for i in range(length)]
