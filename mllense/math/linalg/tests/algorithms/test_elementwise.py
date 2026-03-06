# ==============================
# File: linalg/tests/algorithms/test_elementwise.py
# ==============================
"""Tests for element-wise algorithms."""

from mllense.math.linalg.algorithms.elementwise.add import ElementwiseAdd
from mllense.math.linalg.algorithms.elementwise.multiply import ElementwiseMultiply
from mllense.math.linalg.algorithms.elementwise.divide import ElementwiseDivide
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
from mllense.math.linalg.exceptions import NumericalInstabilityError
import pytest

def test_elementwise_add():
    a = [[1.0, 2.0]]
    b = [[3.0, 4.0]]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    result = ElementwiseAdd().execute(a, b, context=ctx, trace=trace)
    assert result == [[4.0, 6.0]]

def test_elementwise_multiply():
    a = [[2.0, 3.0]]
    b = [[4.0, 5.0]]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    result = ElementwiseMultiply().execute(a, b, context=ctx, trace=trace)
    assert result == [[8.0, 15.0]]

def test_elementwise_divide():
    a = [[10.0, 8.0]]
    b = [[2.0, 4.0]]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    result = ElementwiseDivide().execute(a, b, context=ctx, trace=trace)
    assert result == [[5.0, 2.0]]

def test_elementwise_divide_by_zero():
    a = [[10.0]]
    b = [[0.0]]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    with pytest.raises(NumericalInstabilityError):
        ElementwiseDivide().execute(a, b, context=ctx, trace=trace)
