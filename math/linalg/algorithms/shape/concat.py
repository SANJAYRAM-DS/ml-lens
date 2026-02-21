# ==============================
# File: linalg/algorithms/shape/concat.py
# ==============================
"""Matrix concatenation (horizontal and vertical)."""

from __future__ import annotations

from typing import Any, List

from mllense.math.linalg.algorithms.shape.base import BaseShape
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.exceptions import ShapeMismatchError, InvalidInputError

__all__ = ["ConcatVertical", "ConcatHorizontal"]


class ConcatVertical(BaseShape):
    """Vertical (row-wise) concatenation of matrices."""

    metadata = AlgorithmMetadata(
        name="concat_vertical",
        operation="vstack",
        complexity="O(total_elements)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Vertically stack matrices (all must have same column count).",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        matrices: List[InternalMatrix] = list(args)
        if not matrices:
            raise InvalidInputError("No matrices provided for concatenation.")

        cols = len(matrices[0][0]) if matrices[0] else 0
        for idx, m in enumerate(matrices):
            m_cols = len(m[0]) if m else 0
            if m_cols != cols:
                raise ShapeMismatchError(
                    expected=f"{cols} columns",
                    got=f"{m_cols} columns (matrix {idx})",
                    operation="vstack",
                )

        trace.record(
            operation="vstack",
            description=f"Vertically stacking {len(matrices)} matrices",
        )

        result: InternalMatrix = []
        for m in matrices:
            result.extend([row[:] for row in m])
        return result


class ConcatHorizontal(BaseShape):
    """Horizontal (column-wise) concatenation of matrices."""

    metadata = AlgorithmMetadata(
        name="concat_horizontal",
        operation="hstack",
        complexity="O(total_elements)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Horizontally stack matrices (all must have same row count).",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalMatrix:
        matrices: List[InternalMatrix] = list(args)
        if not matrices:
            raise InvalidInputError("No matrices provided for concatenation.")

        rows = len(matrices[0])
        for idx, m in enumerate(matrices):
            if len(m) != rows:
                raise ShapeMismatchError(
                    expected=f"{rows} rows",
                    got=f"{len(m)} rows (matrix {idx})",
                    operation="hstack",
                )

        trace.record(
            operation="hstack",
            description=f"Horizontally stacking {len(matrices)} matrices",
        )

        result: InternalMatrix = []
        for i in range(rows):
            row: list[float] = []
            for m in matrices:
                row.extend(m[i])
            result.append(row)
        return result
