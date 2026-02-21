# ==============================
# File: linalg/tests/api/test_norms.py
# ==============================
"""Tests for norms API."""

from mllense.math.linalg.api.norms import frobenius_norm, spectral_norm, vector_norm
import math

def test_frobenius_norm():
    a = [[1.0, 2.0], [3.0, 4.0]]
    n = frobenius_norm(a)
    assert math.isclose(n, math.sqrt(30.0))

def test_spectral_norm():
    a = [[1.0, 0.0], [0.0, 2.0]]
    n = spectral_norm(a)
    assert math.isclose(n, 2.0)

def test_vector_norm():
    v = [3.0, 4.0]
    n2 = vector_norm(v, ord=2)
    assert math.isclose(n2, 5.0)
    
    n1 = vector_norm(v, ord=1)
    assert math.isclose(n1, 7.0)

    ninf = vector_norm(v, ord=float("inf"))
    assert math.isclose(ninf, 4.0)
