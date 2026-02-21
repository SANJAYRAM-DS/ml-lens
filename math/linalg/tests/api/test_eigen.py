# ==============================
# File: linalg/tests/api/test_eigen.py
# ==============================
"""Tests for eigen API."""

from mllense.math.linalg.api.eigen import dominant_eigen
import math

def test_dominant_eigen():
    a = [[2.0, 1.0], [1.0, 2.0]]
    val, vec = dominant_eigen(a)
    assert math.isclose(val, 3.0, abs_tol=1e-4)
    assert len(vec) == 2
