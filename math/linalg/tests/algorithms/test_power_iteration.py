# ==============================
# File: linalg/tests/algorithms/test_power_iteration.py
# ==============================
"""Tests for power iteration algorithm."""

from mllense.math.linalg.algorithms.eigen.power_iteration import PowerIteration
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
import math

def test_power_iteration():
    # Symmetric matrix with dominant eigenvalue 3
    a = [[2.0, 1.0], [1.0, 2.0]]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    val, vec = PowerIteration().execute(a, context=ctx, trace=trace, max_iterations=100)
    
    assert math.isclose(val, 3.0, abs_tol=1e-5)
    assert len(vec) == 2
    # The dominant eigenvector is [1/sqrt(2), 1/sqrt(2)]
    # We check if absolute components match because sign could be flipped
    assert math.isclose(abs(vec[0]), 1/math.sqrt(2), abs_tol=1e-5)
    assert math.isclose(abs(vec[1]), 1/math.sqrt(2), abs_tol=1e-5)
