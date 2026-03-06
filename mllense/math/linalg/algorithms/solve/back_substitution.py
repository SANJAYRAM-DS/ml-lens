# ==============================
# File: linalg/algorithms/solve/back_substitution.py
# ==============================
"""Back-substitution for upper-triangular systems."""

from __future__ import annotations

import math
from typing import Any

from mllense.math.linalg._internal.constants import SINGULAR_PIVOT_THRESHOLD
from mllense.math.linalg.algorithms.solve.base import BaseSolve
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.exceptions import SingularMatrixError

__all__ = ["BackSubstitution"]


class BackSubstitution(BaseSolve):
    """Solve ``Ux = b`` where ``U`` is upper-triangular."""

    metadata = AlgorithmMetadata(
        name="back_substitution",
        operation="back_substitution",
        complexity="O(n^2)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Back-substitution for upper-triangular systems Ux = b.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> InternalVector:
        """Solve Ux = b.

        Args:
            args[0]: U (upper-triangular InternalMatrix, n×n)
            args[1]: b (InternalVector, length n)
        """
        u: InternalMatrix = args[0]
        b: InternalVector = args[1]
        n = len(u)

        trace.record(
            operation="back_sub_start",
            description=f"Back-substitution on {n}×{n} upper-triangular system",
        )

        x: InternalVector = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(u[i][i]) < SINGULAR_PIVOT_THRESHOLD:
                raise SingularMatrixError(
                    f"Zero diagonal at U[{i}][{i}]. System is singular."
                )
            s = math.fsum(u[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (b[i] - s) / u[i][i]

        trace.record(
            operation="back_sub_done",
            description=f"Solution computed (length {n})",
            data=x,
        )

        return x
