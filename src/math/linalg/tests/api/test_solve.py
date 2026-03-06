# ==============================
# File: linalg/tests/api/test_solve.py
# ==============================
"""Tests for linear equation solve API."""

from mllense.math.linalg.api.solve import solve
from mllense.math.linalg.exceptions import SingularMatrixError
import pytest
import math

def test_solve_2d():
    a = [[2.0, 1.0], [5.0, 3.0]]
    b = [4.0, 7.0]
    x = solve(a, b)
    assert math.isclose(x[0], 5.0, abs_tol=1e-9)
    assert math.isclose(x[1], -6.0, abs_tol=1e-9)

def test_solve_singular():
    a = [[1.0, 2.0], [2.0, 4.0]]
    b = [1.0, 2.0]
    with pytest.raises(SingularMatrixError):
        solve(a, b)
