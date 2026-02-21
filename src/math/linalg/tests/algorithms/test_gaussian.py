# ==============================
# File: linalg/tests/algorithms/test_gaussian.py
# ==============================
"""Tests for gaussian elimination algorithm."""

from mllense.math.linalg.algorithms.solve.gaussian import GaussianSolve
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.exceptions import SingularMatrixError
import pytest
import math

def test_gaussian_solve():
    a = [[2.0, 1.0], [5.0, 3.0]]
    b = [4.0, 7.0]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    x = GaussianSolve().execute(a, b, context=ctx, trace=trace)
    assert len(x) == 2
    assert math.isclose(x[0], 5.0, abs_tol=1e-9)
    assert math.isclose(x[1], -6.0, abs_tol=1e-9)

def test_gaussian_solve_singular():
    a = [[1.0, 2.0], [2.0, 4.0]]
    b = [1.0, 2.0]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    with pytest.raises(SingularMatrixError):
        GaussianSolve().execute(a, b, context=ctx, trace=trace)
