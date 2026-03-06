# ==============================
# File: linalg/tests/api/test_shape.py
# ==============================
"""Tests for shape manipulation API."""

from mllense.math.linalg.api.shape import reshape, flatten, transpose, vstack, hstack
from mllense.math.linalg.exceptions import ShapeMismatchError
import pytest

def test_reshape():
    a = [[1.0, 2.0, 3.0, 4.0]]
    b = reshape(a, 2, 2)
    assert b == [[1.0, 2.0], [3.0, 4.0]]

def test_reshape_error():
    with pytest.raises(ShapeMismatchError):
        reshape([[1.0, 2.0]], 2, 2)

def test_flatten():
    a = [[1.0, 2.0], [3.0, 4.0]]
    flat = flatten(a)
    assert flat == [1.0, 2.0, 3.0, 4.0]

def test_transpose():
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = transpose(a)
    assert b == [[1.0, 3.0], [2.0, 4.0]]

def test_vstack():
    a = [[1.0, 2.0]]
    b = [[3.0, 4.0]]
    res = vstack(a, b)
    assert res == [[1.0, 2.0], [3.0, 4.0]]

def test_hstack():
    a = [[1.0], [3.0]]
    b = [[2.0], [4.0]]
    res = hstack(a, b)
    assert res == [[1.0, 2.0], [3.0, 4.0]]
