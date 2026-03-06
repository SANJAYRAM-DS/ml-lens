# ==============================
# File: linalg/api/ops.py
# ==============================
"""Public API for element-wise operations: add, subtract, multiply, divide, scalar ops."""

from __future__ import annotations

from mllense.math.linalg.core.metadata import LinalgResult

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


def _format(result: InternalMatrix, return_numpy: bool, algo: Any, ctx: ExecutionContext) -> MatrixLike:
    formatted_val = np.array(result, dtype=np.float64) if return_numpy else result
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if ctx.what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if ctx.how_lense_enabled else "",
        metadata=getattr(algo, "metadata", None)
    )


def add(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Element-wise addition of two matrices."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ElementwiseAdd()
    result = algo.execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy, algo, ctx)


def subtract(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Element-wise subtraction: ``A - B``."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ElementwiseSubtract()
    result = algo.execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy, algo, ctx)


def multiply(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Element-wise (Hadamard) multiplication."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ElementwiseMultiply()
    result = algo.execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy, algo, ctx)


def divide(
    a: MatrixLike,
    b: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Element-wise division: ``A / B``."""
    return_numpy = is_numpy(a) or is_numpy(b)
    a_int = to_internal_matrix(a)
    b_int = to_internal_matrix(b)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ElementwiseDivide()
    result = algo.execute(a_int, b_int, context=ctx, trace=trace)
    return _format(result, return_numpy, algo, ctx)


def scalar_multiply(
    m: MatrixLike,
    scalar: float,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Multiply every element of a matrix by a scalar."""
    return_numpy = is_numpy(m)
    m_int = to_internal_matrix(m)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ScalarMultiply()
    result = algo.execute(m_int, scalar, context=ctx, trace=trace)
    return _format(result, return_numpy, algo, ctx)


def scalar_add(
    m: MatrixLike,
    scalar: float,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Add a scalar to every element of a matrix."""
    return_numpy = is_numpy(m)
    m_int = to_internal_matrix(m)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ScalarAdd()
    result = algo.execute(m_int, scalar, context=ctx, trace=trace)
    return _format(result, return_numpy, algo, ctx)
