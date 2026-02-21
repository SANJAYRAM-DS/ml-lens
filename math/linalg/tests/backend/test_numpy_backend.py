# ==============================
# File: linalg/tests/backend/test_numpy_backend.py
# ==============================
"""Tests for numpy backend logic directly."""

from mllense.math.linalg.backend.numpy_backend import NumpyBackend
import math
import numpy as np

def test_numpy_matmul():
    be = NumpyBackend()
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    res = be.matmul(a, b)
    # Check if native lists returned directly if supplied lists (per spec)
    assert res == [[19.0, 22.0], [43.0, 50.0]]

def test_numpy_solve():
    be = NumpyBackend()
    a = [[2.0, 1.0], [5.0, 3.0]]
    b = [4.0, 7.0]
    res = be.solve(a, b)
    assert math.isclose(res[0], 5.0, abs_tol=1e-5)
    assert math.isclose(res[1], -6.0, abs_tol=1e-5)
