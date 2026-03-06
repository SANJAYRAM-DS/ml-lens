# ==============================
# File: linalg/tests/core/test_execution_context.py
# ==============================
"""Tests for ExecutionContext model."""

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode

def test_execution_context_immutability():
    ctx = ExecutionContext(backend="numpy", mode=ExecutionMode.FAST)
    assert ctx.backend == "numpy"
    assert ctx.mode == ExecutionMode.FAST
    assert ctx.trace_enabled is False
    assert ctx.algorithm_hint is None
    
    ctx2 = ctx.with_overrides(backend="python", trace_enabled=True)
    assert ctx.backend == "numpy"  # original untouched
    assert ctx.trace_enabled is False
    
    assert ctx2.backend == "python"
    assert ctx2.trace_enabled is True
    assert ctx2.mode == ExecutionMode.FAST
