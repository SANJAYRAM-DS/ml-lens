# ==============================
# File: linalg/tests/nn/test_gradients.py
# ==============================
"""Tests for finite displacement computation and numerical comparisons."""

from mllense.math.linalg.nn.gradients import numerical_gradient
import math

def test_numerical_gradient():
    # f(x, y) = x^2 + y^2
    def f(v):
        return v[0]**2 + v[1]**2
        
    x = [2.0, 3.0]
    # analytical grad mapping is == [2*x, 2*y] -> [4.0, 6.0]
    grad = numerical_gradient(f, x, epsilon=1e-5)
    
    assert math.isclose(grad[0], 4.0, rel_tol=1e-4)
    assert math.isclose(grad[1], 6.0, rel_tol=1e-4)
