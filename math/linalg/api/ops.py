# ==============================
# File: linalg/api/ops.py
# ==============================
"""Public API for element-wise operations: add, subtract, multiply, divide, scalar ops."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import (
    InternalMatrix,
    MatrixLike,
    is_numpy,
    to_internal_matrix,
)
from mllense.math.linalg.algorithms.elementwise.add import ElementwiseAdd
from mllense.math.linalg.algorithms.elementwise.subtract import ElementwiseSubtract
from mllense.math.linalg.algorithms.elementwise.multiply import ElementwiseMultiply
from mllense.math.linalg.algorithms.elementwise.divide import ElementwiseDivide
from mllense.math.linalg.algorithms.elementwise.scalar import ScalarMultiply, ScalarAdd

__all__ = ["add", "subtract", "multiply", "divide", "scalar_multiply", "scalar_add"]


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


def _format(result: InternalMatrix, return_numpy: bool) -> MatrixLike:
    if return_numpy:
        return np.array(result, dtype=np.float64)
    return result


def add(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Element-wise addition of two matrices."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ElementwiseAdd().execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy)


def subtract(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Element-wise subtraction: ``A - B``."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ElementwiseSubtract().execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy)


def multiply(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Element-wise (Hadamard) multiplication."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ElementwiseMultiply().execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy)


def divide(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Element-wise division: ``A / B``."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ElementwiseDivide().execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy)


def scalar_multiply(
    m: MatrixLike,
    scalar: float,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Multiply every element of a matrix by a scalar."""
    return_numpy = is_numpy(m)
    m_int = to_internal_matrix(m)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ScalarMultiply().execute(m_int, scalar, context=ctx, trace=trace)
    return _format(result, return_numpy)


def scalar_add(
    m: MatrixLike,
    scalar: float,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Add a scalar to every element of a matrix."""
    return_numpy = is_numpy(m)
    m_int = to_internal_matrix(m)
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ScalarAdd().execute(m_int, scalar, context=ctx, trace=trace)
    return _format(result, return_numpy)
