# ==============================
# File: linalg/api/decomposition.py
# ==============================
"""Public API for decomposition operations: det, inverse, trace, qr, svd, eig."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import (
    InternalMatrix,
    InternalVector,
    MatrixLike,
    is_numpy,
    to_internal_matrix,
)
from mllense.math.linalg.algorithms.decomposition.det import Determinant
from mllense.math.linalg.algorithms.decomposition.inverse import Inverse
from mllense.math.linalg.algorithms.decomposition.trace import MatrixTrace
from mllense.math.linalg.algorithms.decomposition.qr import QRDecomposition
from mllense.math.linalg.algorithms.decomposition.svd import SVDDecomposition
from mllense.math.linalg.algorithms.decomposition.eig import EigenDecomposition

__all__ = ["det", "inv", "matrix_trace", "qr", "svd", "eig"]


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


def det(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> float:
    """Compute the determinant of a square matrix."""
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    return Determinant().execute(a_int, context=ctx, trace=trace)


def inv(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Compute the inverse of a square matrix."""
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    result = Inverse().execute(a_int, context=ctx, trace=trace)
    if return_numpy:
        return np.array(result, dtype=np.float64)
    return result


def matrix_trace(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> float:
    """Compute the trace (sum of diagonal) of a square matrix."""
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    return MatrixTrace().execute(a_int, context=ctx, trace=trace)


def qr(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> Tuple[MatrixLike, MatrixLike]:
    """Compute QR decomposition ``A = QR``."""
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    q, r = QRDecomposition().execute(a_int, context=ctx, trace=trace)
    if return_numpy:
        return np.array(q, dtype=np.float64), np.array(r, dtype=np.float64)
    return q, r


def svd(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> Tuple[MatrixLike, Any, MatrixLike]:
    """Compute SVD decomposition ``A = U Σ V^T``."""
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    u, sigma, vt = SVDDecomposition().execute(a_int, context=ctx, trace=trace)
    if return_numpy:
        return np.array(u, dtype=np.float64), np.array(sigma, dtype=np.float64), np.array(vt, dtype=np.float64)
    return u, sigma, vt


def eig(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> Tuple[Any, MatrixLike]:
    """Compute eigenvalues and eigenvectors."""
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    eigenvalues, eigenvectors = EigenDecomposition().execute(a_int, context=ctx, trace=trace)
    if return_numpy:
        return np.array(eigenvalues, dtype=np.float64), np.array(eigenvectors, dtype=np.float64)
    return eigenvalues, eigenvectors
