# ==============================
# File: linalg/tests/api/test_matmul.py
# ==============================
"""Tests for matmul API."""

from mllense.math.linalg.api.matmul import matmul
import numpy as np
import pytest
from mllense.math.linalg.exceptions import ShapeMismatchError

def test_matmul_2d():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    res = matmul(a, b)
    assert res == [[19.0, 22.0], [43.0, 50.0]]

def test_matmul_numpy_backend():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    res = matmul(a, b)
    assert isinstance(getattr(res, "value", res), np.ndarray)
    assert np.allclose(res, [[19.0, 22.0], [43.0, 50.0]])

def test_matmul_1d():
    res = matmul([1.0, 2.0], [3.0, 4.0])
    assert res == 11.0

def test_matmul_invalid():
    with pytest.raises(ShapeMismatchError):
        matmul([[1.0, 2.0]], [[1.0, 2.0]])
