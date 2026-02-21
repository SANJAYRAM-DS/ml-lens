# ==============================
# File: linalg/api/norms.py
# ==============================
"""Public API for matrix/vector norms."""

from __future__ import annotations

import math
from typing import Any, Optional, Union

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import (
    MatrixLike,
    VectorLike,
    is_numpy,
    to_internal_matrix,
    to_internal_vector,
)
from mllense.math.linalg.algorithms.norms.frobenius import FrobeniusNorm
from mllense.math.linalg.algorithms.norms.spectral import SpectralNorm

__all__ = ["frobenius_norm", "spectral_norm", "vector_norm"]


def _build_context(
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> ExecutionContext:
    from mllense.math.linalg.config import get_config
    cfg = get_config()
    return ExecutionContext(
        backend=backend or cfg.default_backend,
        mode=ExecutionMode.from_string(mode or cfg.default_mode),
        trace_enabled=trace_enabled if trace_enabled is not None else cfg.trace_enabled,
    )


def frobenius_norm(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> float:
    """Compute the Frobenius norm of a matrix."""
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    return FrobeniusNorm().execute(a_int, context=ctx, trace=trace)


def spectral_norm(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> float:
    """Compute the spectral (2-norm) of a matrix."""
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    return SpectralNorm().execute(a_int, context=ctx, trace=trace)


def vector_norm(
    v: VectorLike,
    *,
    ord: int = 2,
) -> float:
    """Compute the p-norm of a vector.

    Args:
        v: Input vector.
        ord: Order of the norm (1, 2, or inf).
    """
    v_int = to_internal_vector(v)
    if ord == 1:
        return sum(abs(x) for x in v_int)
    elif ord == 2:
        return math.sqrt(math.fsum(x * x for x in v_int))
    elif ord == float("inf"):
        return max(abs(x) for x in v_int)
    else:
        return sum(abs(x) ** ord for x in v_int) ** (1.0 / ord)
