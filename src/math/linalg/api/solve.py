# ==============================
# File: linalg/api/solve.py
# ==============================
"""Public API for solving linear systems ``Ax = b``.

Thin wrapper that:
1. Validates inputs.
2. Builds an execution context.
3. Resolves the algorithm via the registry.
4. Executes the algorithm.
5. Returns the result in the caller's original format.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.core.types import (
    InternalVector,
    MatrixLike,
    VectorLike,
    is_numpy,
    to_internal_matrix,
    to_internal_vector,
)
from mllense.math.linalg.core.validation import validate_dimension_limit
from mllense.math.linalg.registry.algorithm_registry import algorithm_registry

__all__ = ["solve"]


def solve(
    a: MatrixLike,
    b: VectorLike,
    *,
    backend: Optional[str] = None,
    mode: Optional[str] = None,
    algorithm: Optional[str] = None,
    trace_enabled: Optional[bool] = None,
) -> Any:
    """Solve the linear system ``Ax = b`` for ``x``.

    Args:
        a: Coefficient matrix (must be square, n × n).
        b: Right-hand-side vector (length n).
        backend: Override default backend.
        mode: Override default mode.
        algorithm: Explicit algorithm hint (e.g. ``"gaussian"``).
        trace_enabled: Override global trace flag.

    Returns:
        Solution vector ``x`` in the same format as the input.
    """
    # ── detect format ────────────────────────────────────────────────── #
    return_numpy = is_numpy(a) or is_numpy(b)

    # ── normalise ────────────────────────────────────────────────────── #
    a_int = to_internal_matrix(a)
    b_int = to_internal_vector(b)

    rows = len(a_int)
    cols = len(a_int[0]) if rows else 0
    validate_dimension_limit(rows, cols)

    # ── build execution context ──────────────────────────────────────── #
    ctx = _build_context(backend, mode, algorithm, trace_enabled)

    # ── resolve algorithm ────────────────────────────────────────────── #
    algo = algorithm_registry.get("solve", ctx, matrix_dim=rows)

    # ── execute ──────────────────────────────────────────────────────── #
    trace = Trace(enabled=ctx.trace_enabled)
    x: InternalVector = algo.execute(a_int, b_int, context=ctx, trace=trace)

    # ── format result ────────────────────────────────────────────────── #
    if return_numpy:
        return np.array(x, dtype=np.float64)
    return x


# ── private helpers ──────────────────────────────────────────────────────── #

def _build_context(
    backend: Optional[str],
    mode: Optional[str],
    algorithm: Optional[str],
    trace_enabled: Optional[bool],
) -> ExecutionContext:
    from mllense.math.linalg.config import get_config

    cfg = get_config()
    return ExecutionContext(
        backend=backend or cfg.default_backend,
        mode=ExecutionMode.from_string(mode or cfg.default_mode),
        trace_enabled=trace_enabled if trace_enabled is not None else cfg.trace_enabled,
        algorithm_hint=algorithm,
    )