# ==============================
# File: linalg/algorithms/decomposition/det.py
# ==============================
"""Determinant computation via LU decomposition."""

from __future__ import annotations

from typing import Any

from mllense.math.linalg.algorithms.decomposition.base import BaseDecomposition
from mllense.math.linalg.algorithms.solve.lu import lu_decompose
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix
from mllense.math.linalg.core.validation import validate_square
from mllense.math.linalg.exceptions import SingularMatrixError

__all__ = ["Determinant"]


class Determinant(BaseDecomposition):
    """Compute the determinant via LU decomposition."""

    metadata = AlgorithmMetadata(
        name="determinant",
        operation="det",
        complexity="O(n^3)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Compute matrix determinant via LU decomposition.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> float:
        a: InternalMatrix = args[0]
        n = validate_square(a, operation="det")

        trace.record(
            operation="det_start",
            description=f"Computing determinant of {n}×{n} matrix via LU",
        )

        try:
            _l, u, perm = lu_decompose(a, trace=trace)
        except SingularMatrixError:
            trace.record(operation="det_done", description="Determinant = 0 (singular)")
            return 0.0

        # det = product of U diagonal * sign of permutation
        det_val = 1.0
        for i in range(n):
            det_val *= u[i][i]

        # count swaps
        swaps = 0
        perm_copy = perm[:]
        for i in range(n):
            while perm_copy[i] != i:
                j = perm_copy[i]
                perm_copy[i], perm_copy[j] = perm_copy[j], perm_copy[i]
                swaps += 1

        if swaps % 2 == 1:
            det_val = -det_val

        trace.record(
            operation="det_done",
            description=f"Determinant = {det_val}",
        )

        return det_val
