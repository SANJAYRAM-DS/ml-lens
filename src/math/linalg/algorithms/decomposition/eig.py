# ==============================
# File: linalg/algorithms/decomposition/eig.py
# ==============================
"""Eigenvalue decomposition (wraps numpy)."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from mllense.math.linalg.algorithms.decomposition.base import BaseDecomposition
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.core.validation import validate_square

__all__ = ["EigenDecomposition"]


class EigenDecomposition(BaseDecomposition):
    """Compute eigenvalues and eigenvectors of a square matrix.

    Delegates to ``numpy.linalg.eig`` for robustness.
    """

    metadata = AlgorithmMetadata(
        name="eigen_decomposition",
        operation="eig",
        complexity="O(n^3)",
        stable=True,
        supports_batch=False,
        requires_square=True,
        description="Eigenvalue decomposition via numpy.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Tuple[InternalVector, InternalMatrix]:
        """Compute eigenvalues and eigenvectors.

        Returns:
            (eigenvalues, eigenvectors) where eigenvectors are column-wise.
        """
        a: InternalMatrix = args[0]
        n = validate_square(a, operation="eig")

        trace.record(
            operation="eig_start",
            description=f"Eigendecomposition of {n}×{n} matrix",
        )

        a_np = np.array(a, dtype=np.float64)
        eigenvalues_np, eigenvectors_np = np.linalg.eig(a_np)

        eigenvalues: InternalVector = eigenvalues_np.real.tolist()
        eigenvectors: InternalMatrix = eigenvectors_np.real.tolist()

        trace.record(
            operation="eig_done",
            description=f"Found {len(eigenvalues)} eigenvalues",
            data={"eigenvalues": eigenvalues},
        )

        return eigenvalues, eigenvectors
