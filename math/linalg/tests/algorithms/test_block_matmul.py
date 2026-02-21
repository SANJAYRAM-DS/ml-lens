# ==============================
# File: linalg/tests/algorithms/test_block_matmul.py
# ==============================
"""Tests for block matmul algorithm."""

from mllense.math.linalg.algorithms.matmul.block import BlockMatmul
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace
import math

def test_block_matmul_2d_2d():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    result = BlockMatmul().execute(a, b, context=ctx, trace=trace, block_size=1)
    
    assert len(result) == 2
    assert result[0] == [19.0, 22.0]
    assert result[1] == [43.0, 50.0]

def test_block_matmul_1d_1d():
    a = [[1.0, 2.0, 3.0]]
    b = [[4.0], [5.0], [6.0]]
    ctx = ExecutionContext("python", ExecutionMode.FAST, False)
    trace = Trace(False)
    
    result = BlockMatmul().execute(a, b, context=ctx, trace=trace)
    
    assert isinstance(result, float)
    assert math.isclose(result, 32.0)
