# ==============================
# File: linalg/algorithms/decomposition/svd.py
# ==============================
"""Singular Value Decomposition (SVD).

Wraps numpy for the heavy lifting but provides the engine's
standard interface and trace integration.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from mllense.math.linalg.algorithms.decomposition.base import BaseDecomposition
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, InternalVector

__all__ = ["SVDDecomposition"]


class SVDDecomposition(BaseDecomposition):
    """Compute the Singular Value Decomposition ``A = U Σ V^T``.

    Delegates to ``numpy.linalg.svd`` for numerical robustness,
    but wraps the result in the engine's internal format.
    """

    metadata = AlgorithmMetadata(
        name="svd_decomposition",
        operation="svd",
        complexity="O(min(mn^2, m^2n))",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Singular Value Decomposition via numpy (or power-iteration fallback).",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Tuple[InternalMatrix, InternalVector, InternalMatrix]:
        """Compute SVD.

        Args:
            args[0]: A (InternalMatrix m×n)

        Returns:
            (U, sigma, Vt) where sigma is the list of singular values.
        """
        a: InternalMatrix = args[0]
        m = len(a)
        n = len(a[0]) if m else 0

        trace.record(
            operation="svd_start",
            description=f"SVD of {m}×{n} matrix",
        )

        a_np = np.array(a, dtype=np.float64)
        u_np, s_np, vt_np = np.linalg.svd(a_np, full_matrices=True)

        u: InternalMatrix = u_np.tolist()
        sigma: InternalVector = s_np.tolist()
        vt: InternalMatrix = vt_np.tolist()

        trace.record(
            operation="svd_done",
            description=f"U: {len(u)}×{len(u[0]) if u else 0}, "
                        f"sigma: {len(sigma)} values, "
                        f"Vt: {len(vt)}×{len(vt[0]) if vt else 0}",
        )

        return u, sigma, vt
