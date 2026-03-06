# ==============================
# File: linalg/api/matmul.py
# ==============================
"""Public API for matrix multiplication.

Thin wrapper that:
1. Validates inputs.
2. Builds an execution context.
3. Resolves the algorithm via the registry.
4. Executes the algorithm.
5. Returns the result in the caller's original format.
"""

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
    VectorLike,
    is_numpy,
    to_internal_matrix,
    to_internal_vector,
)
from mllense.math.linalg.core.validation import validate_dimension_limit
from mllense.math.linalg.exceptions import InvalidInputError
from mllense.math.linalg.registry.algorithm_registry import algorithm_registry
from mllense.math.linalg.registry.backend_registry import backend_registry

__all__ = ["matmul"]


def matmul(
    a: Union[MatrixLike, VectorLike],
    b: Union[MatrixLike, VectorLike],
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    algorithm: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
    what_lense: bool = True,
    how_lense: bool = False,
) -> Any:
    """Multiply two matrices / vectors, behaving like ``numpy.matmul``.

    Supported shapes:

    * **1D × 1D**  → scalar (inner product)
    * **2D × 1D**  → 1D vector
    * **2D × 2D**  → 2D matrix

    Args:
        a: Left operand (matrix or vector).
        b: Right operand (matrix or vector).
        backend: Override default backend (``"numpy"`` / ``"python"``).
        mode: Override default mode (``"fast"`` / ``"educational"`` / ``"debug"``).
        algorithm: Explicit algorithm hint (e.g. ``"naive"``).
        trace_enabled: Override global trace flag.

    Returns:
        The product, in the same format as the input (ndarray if input was
        ndarray, list if input was list).
    """
    # ── detect input format ──────────────────────────────────────────── #
    return_numpy = is_numpy(a) or is_numpy(b)
    a_is_1d, b_is_1d = _is_1d(a), _is_1d(b)

    # ── normalise to 2-D internal format ─────────────────────────────── #
    a_int = _to_2d(a, "a")
    b_int = _to_2d(b, "b")

    # dimension limit
    a_rows, a_cols = len(a_int), len(a_int[0])
    b_rows, b_cols = len(b_int), len(b_int[0])
    validate_dimension_limit(a_rows, a_cols)
    validate_dimension_limit(b_rows, b_cols)

    # ── build execution context ──────────────────────────────────────── #
    ctx = _build_context(backend, mode, algorithm, trace_enabled, what_lense_enabled=what_lense, how_lense_enabled=how_lense)

    # ── resolve algorithm ────────────────────────────────────────────── #
    max_dim = max(a_rows, a_cols, b_rows, b_cols)
    algo = algorithm_registry.get("matmul", ctx, matrix_dim=max_dim)

    # ── execute ──────────────────────────────────────────────────────── #
    trace = Trace(enabled=ctx.trace_enabled)
    raw_result = algo.execute(a_int, b_int, context=ctx, trace=trace)

    # ── format result ────────────────────────────────────────────────── #
    formatted_val = _format_result(raw_result, return_numpy, a_is_1d, b_is_1d)
    return LinalgResult(
        value=formatted_val,
        what_lense=algo._generate_what_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "what_lense_enabled")) and locals()["ctx"].what_lense_enabled else "",
        how_lense=algo._finalize_how_lense() if "algo" in locals() else "" if ("ctx" in locals() and hasattr(locals()["ctx"], "how_lense_enabled")) and locals()["ctx"].how_lense_enabled else "",
        metadata=getattr(locals().get("algo"), "metadata", None)
    )


# ── private helpers ──────────────────────────────────────────────────────── #

def _is_1d(x: Any) -> bool:
    """Check if the original user input is 1-D."""
    if isinstance(x, np.ndarray):
        return x.ndim == 1
    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return True
        return not isinstance(x[0], (list, tuple, np.ndarray))
    return False


def _to_2d(x: Any, label: str) -> InternalMatrix:
    """Convert user input to a 2-D internal matrix.

    * 1-D inputs become a *row* vector ``[[x0, x1, ...]]`` for the left
      operand and a *column* vector ``[[x0], [x1], ...]`` for the right
      operand.  The API layer handles shape semantics (matching numpy).
    """
    if _is_1d(x):
        vec = to_internal_vector(x)
        if label == "a":
            # left 1-D → row vector (1×n)
            return [vec]
        else:
            # right 1-D → column vector (n×1)
            return [[v] for v in vec]
    return to_internal_matrix(x)


def _build_context(
    backend: Optional[str],
    mode: Optional[str],
    algorithm: Optional[str],
    trace_enabled: Optional[bool],
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
        
        algorithm_hint=algorithm,
    )


def _format_result(
    raw: Any,
    return_numpy: bool,
    a_is_1d: bool,
    b_is_1d: bool,
) -> Any:
    """Convert internal result back to the caller's expected format."""
    # scalar
    if isinstance(raw, (int, float)):
        return float(raw) if not return_numpy else np.float64(raw)

    # 1-D vector (list)
    if isinstance(raw, list) and raw and not isinstance(raw[0], list):
        if return_numpy:
            return np.array(raw, dtype=np.float64)
        return raw

    # 2-D matrix (list of lists)
    if isinstance(raw, list) and raw and isinstance(raw[0], list):
        if return_numpy:
            return np.array(raw, dtype=np.float64)
        return raw

    return raw
