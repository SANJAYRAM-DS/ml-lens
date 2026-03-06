# ==============================
# File: linalg/algorithms/eigen/power_iteration.py
# ==============================
"""Power iteration for finding the dominant eigenvalue/eigenvector."""

from __future__ import annotations

import math
from typing import Any, Tuple

from mllense.math.linalg.algorithms.eigen.base import BaseEigen
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_square
from mllense.math.linalg.exceptions import NumericalInstabilityError

__all__ = ["PowerIteration"]

_MAX_ITERATIONS = 1000
_CONVERGENCE_TOL = 1e-10


class PowerIteration(BaseEigen):
    """Find the dominant eigenvalue and eigenvector via power iteration."""

    metadata = AlgorithmMetadata(
        name="power_iteration",
        operation="dominant_eigen",
        complexity="O(n^2 * iterations)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Power iteration for dominant eigenvalue/eigenvector.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Tuple[float, InternalVector]:
        """Find dominant eigenvalue and eigenvector.

        Args:
            args[0]: A (square matrix)

        Keyword Args:
            max_iterations: Max iterations (default 1000).
            tolerance: Convergence tolerance (default 1e-10).

        Returns:
            (eigenvalue, eigenvector)
        """
        a: InternalMatrix = args[0]
        n = validate_square(a, operation="power_iteration")
        max_iter: int = kwargs.get("max_iterations", _MAX_ITERATIONS)
        tol: float = kwargs.get("tolerance", _CONVERGENCE_TOL)

        trace.record(
            operation="power_iter_start",
            description=f"Power iteration on {n}×{n} matrix, max_iter={max_iter}",
        )

        # initial vector: [1, 1, ..., 1] normalized
        b: InternalVector = [1.0 / math.sqrt(n)] * n
        eigenvalue = 0.0

        for iteration in range(max_iter):
            # matrix-vector multiply: Ab
            ab: InternalVector = [0.0] * n
            for i in range(n):
                ab[i] = math.fsum(a[i][j] * b[j] for j in range(n))

            # compute eigenvalue estimate (Rayleigh quotient)
            new_eigenvalue = math.fsum(ab[i] * b[i] for i in range(n))

            # normalize
            norm = math.sqrt(math.fsum(x * x for x in ab))
            if norm < 1e-15:
                raise NumericalInstabilityError(
                    f"Near-zero vector norm at iteration {iteration}. "
                    f"Matrix may be zero or degenerate."
                )
            b = [x / norm for x in ab]

            # convergence check
            if abs(new_eigenvalue - eigenvalue) < tol:
                trace.record(
                    operation="power_iter_converged",
                    description=f"Converged at iteration {iteration + 1}, "
                                f"eigenvalue = {new_eigenvalue}",
                )
                return new_eigenvalue, b

            eigenvalue = new_eigenvalue

        trace.record(
            operation="power_iter_done",
            description=f"Max iterations reached. Best eigenvalue = {eigenvalue}",
        )

        return eigenvalue, b
