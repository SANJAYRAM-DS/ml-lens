# ==============================
# File: linalg/tests/api/test_ops.py
# ==============================
"""Tests for general ops API."""

from mllense.math.linalg.api.ops import add, subtract, multiply, divide, scalar_multiply, scalar_add

def test_add():
    res = add([[1.0]], [[2.0]])
    assert res == [[3.0]]

def test_subtract():
    res = subtract([[3.0]], [[2.0]])
    assert res == [[1.0]]

def test_multiply():
    res = multiply([[2.0, 3.0]], [[4.0, 5.0]])
    assert res == [[8.0, 15.0]]

def test_divide():
    res = divide([[10.0]], [[2.0]])
    assert res == [[5.0]]

def test_scalar_multiply():
    res = scalar_multiply([[2.0]], 5.0)
    assert res == [[10.0]]

def test_scalar_add():
    res = scalar_add([[2.0]], 5.0)
    assert res == [[7.0]]
