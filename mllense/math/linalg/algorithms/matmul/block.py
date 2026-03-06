# ==============================
# File: linalg/algorithms/matmul/block.py
# ==============================
"""Block (tiled) matrix multiplication for improved cache locality.

Complexity: O(m*k*n) — same as naive but with better cache behaviour
for large matrices due to blocking.
"""

from __future__ import annotations

from typing import Any, Union

from mllense.math.linalg._internal.constants import FLOAT_OVERFLOW_GUARD
from mllense.math.linalg.algorithms.matmul.base import BaseMatmul
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_matmul_shapes
from mllense.math.linalg.exceptions import NumericalInstabilityError

__all__ = ["BlockMatmul"]

DEFAULT_BLOCK_SIZE = 64


class BlockMatmul(BaseMatmul):
    """Block (tiled) matrix multiplication."""

    metadata = AlgorithmMetadata(
        name="block_matmul",
        operation="matmul",
        complexity="O(m*k*n)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description=(
            "Block (tiled) matrix multiplication.  Same asymptotic complexity "
            "as naive O(n³) but with improved cache locality for large matrices."
        ),
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Union[InternalMatrix, InternalVector, float]:
        a: InternalMatrix = args[0]
        b: InternalMatrix = args[1]
        block_size: int = kwargs.get("block_size", DEFAULT_BLOCK_SIZE)

        a_rows = len(a)
        a_cols = len(a[0]) if a_rows else 0
        b_rows = len(b)
        b_cols = len(b[0]) if b_rows else 0

        m, k, n = validate_matmul_shapes((a_rows, a_cols), (b_rows, b_cols))

        trace.record(
            operation="block_matmul_start",
            description=f"Block matmul: ({m}×{k}) @ ({k}×{n}), block_size={block_size}",
        )

        result: InternalMatrix = [[0.0] * n for _ in range(m)]

        for ii in range(0, m, block_size):
            for jj in range(0, n, block_size):
                for kk in range(0, k, block_size):
                    i_end = min(ii + block_size, m)
                    j_end = min(jj + block_size, n)
                    k_end = min(kk + block_size, k)
                    for i in range(ii, i_end):
                        for k_idx in range(kk, k_end):
                            a_val = a[i][k_idx]
                            if a_val == 0.0:
                                continue
                            for j in range(jj, j_end):
                                result[i][j] += a_val * b[k_idx][j]
                                if abs(result[i][j]) > FLOAT_OVERFLOW_GUARD:
                                    raise NumericalInstabilityError(
                                        f"Float overflow at result[{i}][{j}]"
                                    )

        trace.record(
            operation="block_matmul_done",
            description=f"Result shape: ({m}×{n})",
            data=result,
        )

        if m == 1 and n == 1:
            return result[0][0]
        if n == 1:
            return [row[0] for row in result]
        if m == 1:
            return result[0]
        return result
