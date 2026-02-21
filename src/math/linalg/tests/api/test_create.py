# ==============================
# File: linalg/tests/api/test_create.py
# ==============================
"""Tests for the create API (zeros, ones, eye, rand)."""

from mllense.math.linalg.api.create import zeros, ones, eye, rand
from mllense.math.linalg.exceptions import InvalidInputError
import pytest
import numpy as np

def test_zeros():
    res = zeros(2, 3)
    assert res == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    
def test_zeros_numpy():
    res = zeros(2, as_numpy=True)
    assert isinstance(res, np.ndarray)
    assert np.allclose(res, np.zeros((2, 2)))

def test_ones():
    res = ones(2)
    assert res == [[1.0, 1.0], [1.0, 1.0]]

def test_eye():
    res = eye(2)
    assert res == [[1.0, 0.0], [0.0, 1.0]]

def test_rand_seed():
    r1 = rand(2, seed=42)
    r2 = rand(2, seed=42)
    assert r1 == r2
    assert len(r1) == 2 and len(r1[0]) == 2
    for row in r1:
        for v in row:
            assert 0.0 <= v < 1.0

def test_negative_dims():
    with pytest.raises(InvalidInputError):
        zeros(0, 5)
    with pytest.raises(InvalidInputError):
        ones(5, -1)
