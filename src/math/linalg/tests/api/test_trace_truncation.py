import pytest
from mllense.math.linalg import matmul, rand
from mllense.math.linalg.core.metadata import LinalgResult

def test_truncation_logic():
    # If steps > 20 (we use 10 for limit per user prompt: keep first 5, last 5)
    # A 5x5 matrix math will do 5*5*5 = 125 steps for triple loop.
    a = rand(5, 5, seed=1)
    b = rand(5, 5, seed=2)
    
    result = matmul(a, b, how_lense=True)
    how = result.how_lense
    
    assert "intermediate" in how
    
    lines = how.split("\n")
    # Verify first five and last five
    # It should not be massive
    assert len(lines) < 30
