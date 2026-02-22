# ==============================
# File: linalg/api/shape.py
# ==============================
"""Public API for shape-manipulation operations."""

from __future__ import annotations

from mllense.math.linalg.core.metadata import LinalgResult

from typing import Any, List, Optional, Union

import numpy as np

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import (
    InternalMatrix,
    InternalVector,
    MatrixLike,
    VectorLike,
    is_numpy,
    to_internal_matrix,
    to_internal_vector,
)
from mllense.math.linalg.algorithms.shape.reshape import Reshape
from mllense.math.linalg.algorithms.shape.flatten import Flatten
from mllense.math.linalg.algorithms.shape.concat import ConcatVertical, ConcatHorizontal
from mllense.math.linalg.algorithms.matmul.transpose import Transpose

__all__ = ["reshape", "flatten", "transpose", "vstack", "hstack"]


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


def reshape(
    a: MatrixLike,
    new_rows: int,
    new_cols: int,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Reshape a matrix to ``(new_rows, new_cols)``."""
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    result = Reshape().execute(a_int, new_rows, new_cols, context=ctx, trace=trace)
    formatted_val = np.array(result, dtype=np.float64) if return_numpy else result
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


def flatten(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> VectorLike:
    """Flatten a matrix to a 1-D vector."""
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    result = Flatten().execute(a_int, context=ctx, trace=trace)
    formatted_val = np.array(result, dtype=np.float64) if return_numpy else result
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


def transpose(
    a: MatrixLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Transpose a matrix."""
    return_numpy = is_numpy(a)
    a_int = to_internal_matrix(a)
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    result = Transpose().execute(a_int, context=ctx, trace=trace)
    formatted_val = np.array(result, dtype=np.float64) if return_numpy else result
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


def vstack(
    *matrices: MatrixLike,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Vertically stack matrices."""
    return_numpy = any(is_numpy(m) for m in matrices)
    internals = [to_internal_matrix(m) for m in matrices]
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ConcatVertical().execute(*internals, context=ctx, trace=trace)
    formatted_val = np.array(result, dtype=np.float64) if return_numpy else result
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


def hstack(
    *matrices: MatrixLike,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> MatrixLike:
    """Horizontally stack matrices."""
    return_numpy = any(is_numpy(m) for m in matrices)
    internals = [to_internal_matrix(m) for m in matrices]
    ctx = _build_context(backend, mode, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)
    trace = Trace(enabled=ctx.trace_enabled)
    result = ConcatHorizontal().execute(*internals, context=ctx, trace=trace)
    formatted_val = np.array(result, dtype=np.float64) if return_numpy else result
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )
