# ==============================
# File: linalg/algorithms/decomposition/qr.py
# ==============================
"""QR decomposition via Modified Gram-Schmidt."""

from __future__ import annotations

import math
from typing import Any, Tuple

from mllense.math.linalg.algorithms.decomposition.base import BaseDecomposition
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.core.validation import validate_square
from mllense.math.linalg.exceptions import NumericalInstabilityError

__all__ = ["QRDecomposition"]


class QRDecomposition(BaseDecomposition):
    """QR decomposition using Modified Gram-Schmidt.

    Decomposes ``A = Q R`` where Q is orthogonal and R is upper-triangular.
    """

    metadata = AlgorithmMetadata(
        name="qr_decomposition",
        operation="qr",
        complexity="O(2mn^2)",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="QR decomposition via Modified Gram-Schmidt process.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Tuple[InternalMatrix, InternalMatrix]:
        """Compute QR decomposition.

        Args:
            args[0]: A (InternalMatrix m×n, m >= n)

        Returns:
            (Q, R) tuple.
        """
        a: InternalMatrix = args[0]
        m = len(a)
        n = len(a[0]) if m else 0

        trace.record(
            operation="qr_start",
            description=f"QR decomposition of {m}×{n} matrix (Modified Gram-Schmidt)",
        )

        # work with column-major for convenience
        # columns[j] = list of m floats
        columns: list[list[float]] = [
            [a[i][j] for i in range(m)] for j in range(n)
        ]

        q_cols: list[list[float]] = []
        r: InternalMatrix = [[0.0] * n for _ in range(n)]

        for j in range(n):
            v = columns[j][:]

            for i in range(len(q_cols)):
                # r[i][j] = <q_i, v>
                r_ij = math.fsum(q_cols[i][k] * v[k] for k in range(m))
                r[i][j] = r_ij
                # v = v - r_ij * q_i
                for k in range(m):
                    v[k] -= r_ij * q_cols[i][k]

            # r[j][j] = ||v||
            norm_v = math.sqrt(math.fsum(x * x for x in v))
            if norm_v < 1e-14:
                raise NumericalInstabilityError(
                    f"Near-zero column norm at column {j}. "
                    f"Matrix may be rank-deficient."
                )
            r[j][j] = norm_v

            # q_j = v / ||v||
            q_j = [x / norm_v for x in v]
            q_cols.append(q_j)

        # convert Q from column-major to row-major
        q: InternalMatrix = [
            [q_cols[j][i] for j in range(n)] for i in range(m)
        ]

        trace.record(
            operation="qr_done",
            description=f"Q is {m}×{n}, R is {n}×{n}",
            data={"Q_shape": (m, n), "R_shape": (n, n)},
        )

        return q, r
