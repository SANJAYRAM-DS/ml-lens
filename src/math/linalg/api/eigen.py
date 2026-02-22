# ==============================
# File: linalg/api/eigen.py
# ==============================
"""Public API for eigenvalue operations."""

from __future__ import annotations

from mllense.math.linalg.core.metadata import LinalgResult

from typing import Any, Optional, Tuple

import numpy as np

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalVector, MatrixLike, is_numpy, to_internal_matrix
from mllense.math.linalg.algorithms.eigen.power_iteration import PowerIteration

__all__ = ["dominant_eigen"]


def _build_context(
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense_enabled: bool = True,
    how_lense_enabled: bool = False,
) -> ExecutionContext:
    from mllense.math.linalg.config import get_config
    cfg = get_config()
    return ExecutionContext(
        backend=backend or cfg.default_backend,
        mode=ExecutionMode.from_string(mode or cfg.default_mode),
        trace_enabled=trace_enabled if trace_enabled is not None else cfg.trace_enabled,
        what_lense_enabled=what_lense_enabled,
        how_lense_enabled=how_lense_enabled,
        
    )


def dominant_eigen(
    a: MatrixLike,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-10,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> Tuple[float, Any]:
    """Find the dominant eigenvalue and eigenvector via power iteration.

    Returns:
        (eigenvalue, eigenvector)
    """
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    eigenvalue, eigenvector = PowerIteration().execute(
        a_int, context=ctx, trace=trace,
        max_iterations=max_iterations, tolerance=tolerance,
    )
    if return_numpy:
        return eigenvalue, np.array(eigenvector, dtype=np.float64)
    return eigenvalue, eigenvector
