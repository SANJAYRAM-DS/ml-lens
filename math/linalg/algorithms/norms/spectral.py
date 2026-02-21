# ==============================
# File: linalg/algorithms/norms/spectral.py
# ==============================
"""Spectral norm: largest singular value."""

from __future__ import annotations

from typing import Any

import numpy as np

from mllense.math.linalg.algorithms.norms.base import BaseNorm
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix

__all__ = ["SpectralNorm"]


class SpectralNorm(BaseNorm):
    """Spectral norm (2-norm): largest singular value of A."""

    metadata = AlgorithmMetadata(
        name="spectral_norm",
        operation="norm_spectral",
        complexity="O(min(mn^2, m^2n))",
        stable=True,
        supports_batch=False,
        requires_square=False,
        description="Spectral (operator 2-norm) — largest singular value.",
    )

    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> float:
        m: InternalMatrix = args[0]

        trace.record(
            operation="spectral_start",
            description="Computing spectral norm (largest singular value)",
        )

        a_np = np.array(m, dtype=np.float64)
        s = np.linalg.svd(a_np, compute_uv=False)
        result = float(s[0]) if len(s) > 0 else 0.0

        trace.record(
            operation="spectral_done",
            description=f"Spectral norm = {result}",
        )

        return result
