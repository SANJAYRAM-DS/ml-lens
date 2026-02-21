# ==============================
# File: linalg/tests/backend/test_python_backend.py
# ==============================
"""Tests for python backend logic directly."""

from mllense.math.linalg.backend.python_backend import PythonBackend
import math
from mllense.math.linalg.exceptions import SingularMatrixError
import pytest

def test_python_matmul():
    be = PythonBackend()
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    res = be.matmul(a, b)
    assert res == [[19.0, 22.0], [43.0, 50.0]]

def test_python_solve():
    be = PythonBackend()
    a = [[2.0, 1.0], [5.0, 3.0]]
    b = [4.0, 7.0]
    res = be.solve(a, b)
    assert math.isclose(res[0], 5.0, abs_tol=1e-5)
    assert math.isclose(res[1], -6.0, abs_tol=1e-5)

def test_python_solve_singular():
    be = PythonBackend()
    a = [[1.0, 2.0], [2.0, 4.0]]
    b = [1.0, 2.0]
    with pytest.raises(SingularMatrixError):
        be.solve(a, b)
