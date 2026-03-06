# ==============================
# File: linalg/api/create.py
# ==============================
"""Public API for matrix creation operations: zeros, ones, eye, rand."""

from __future__ import annotations

from mllense.math.linalg.core.metadata import LinalgResult

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
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Create a zero matrix of shape ``(rows, cols)``.

    Args:
        rows: Number of rows.
        cols: Number of columns (defaults to ``rows``).
        as_numpy: If True, return a numpy array.
    """
    c = cols if cols is not None else rows
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = ZerosCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace)
    formatted_val = _format_result(result, as_numpy=as_numpy)
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


def ones(
    rows: int,
    cols: Optional[int] = None,
    *,
    as_numpy: bool = False,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Create a matrix of ones of shape ``(rows, cols)``.

    Args:
        rows: Number of rows.
        cols: Number of columns (defaults to ``rows``).
        as_numpy: If True, return a numpy array.
    """
    c = cols if cols is not None else rows
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = OnesCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace)
    formatted_val = _format_result(result, as_numpy=as_numpy)
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


def eye(
    rows: int,
    cols: Optional[int] = None,
    *,
    as_numpy: bool = False,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Create an identity-like matrix of shape ``(rows, cols)``.

    Args:
        rows: Number of rows.
        cols: Number of columns (defaults to ``rows``).
        as_numpy: If True, return a numpy array.
    """
    c = cols if cols is not None else rows
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = EyeCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace)
    formatted_val = _format_result(result, as_numpy=as_numpy)
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


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
    what_lense: bool = True,
    how_lense: bool = False,
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
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    algo = RandCreation()
    result = algo.execute(rows, c, context=ctx, trace=trace, seed=seed, low=low, high=high)
    formatted_val = _format_result(result, as_numpy=as_numpy)
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )
