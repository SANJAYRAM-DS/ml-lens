# ==============================
# File: linalg/api/create.py
# ==============================
"""Public API for matrix creation operations: zeros, ones, eye, rand."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import InternalMatrix, MatrixLike
from mllense.math.linalg.exceptions import InvalidInputError

# Direct algorithm imports (creation is not registry-routed by default)
from mllense.math.linalg.algorithms.creation.zeros import ZerosCreation
from mllense.math.linalg.algorithms.creation.ones import OnesCreation
from mllense.math.linalg.algorithms.creation.eye import EyeCreation
from mllense.math.linalg.algorithms.creation.rand import RandCreation

__all__ = ["zeros", "ones", "eye", "rand"]


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


def _format_result(
    internal: InternalMatrix, *, as_numpy: bool = False
) -> MatrixLike:
    if as_numpy:
        return np.array(internal, dtype=np.float64)
    return internal


def zeros(
    rows: int,
    cols: Optional[int] = None,
    *,
    as_numpy: bool = False,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Create a zero matrix of shape ``(rows, cols)``.

    Args:
        rows: Number of rows.
        cols: Number of columns (defaults to ``rows``).
        as_numpy: If True, return a numpy array.
    """
    c = cols if cols is not None else rows
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ZerosCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace)
    return _format_result(result, as_numpy=as_numpy)


def ones(
    rows: int,
    cols: Optional[int] = None,
    *,
    as_numpy: bool = False,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Create a matrix of ones of shape ``(rows, cols)``.

    Args:
        rows: Number of rows.
        cols: Number of columns (defaults to ``rows``).
        as_numpy: If True, return a numpy array.
    """
    c = cols if cols is not None else rows
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = OnesCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace)
    return _format_result(result, as_numpy=as_numpy)


def eye(
    rows: int,
    cols: Optional[int] = None,
    *,
    as_numpy: bool = False,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Create an identity-like matrix of shape ``(rows, cols)``.

    Args:
        rows: Number of rows.
        cols: Number of columns (defaults to ``rows``).
        as_numpy: If True, return a numpy array.
    """
    c = cols if cols is not None else rows
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = EyeCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace)
    return _format_result(result, as_numpy=as_numpy)


def rand(
    rows: int,
    cols: Optional[int] = None,
    *,
    seed: Optional[int] = None,
    low: float = 0.0,
    high: float = 1.0,
    as_numpy: bool = False,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> MatrixLike:
    """Create a random matrix of shape ``(rows, cols)``.

    Args:
        rows: Number of rows.
        cols: Number of columns (defaults to ``rows``).
        seed: Random seed for reproducibility.
        low: Lower bound for values.
        high: Upper bound for values.
        as_numpy: If True, return a numpy array.
    """
    c = cols if cols is not None else rows
    ctx = _build_context(backend, mode, trace_enabled)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = RandCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace, seed=seed, low=low, high=high)
    return _format_result(result, as_numpy=as_numpy)
