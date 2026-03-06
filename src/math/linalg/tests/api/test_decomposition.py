# ==============================
# File: linalg/tests/api/test_decomposition.py
# ==============================
"""Tests for the decomposition API (det, inv, qr, svd, eig)."""

from mllense.math.linalg.api.decomposition import det, inv, matrix_trace, qr, svd, eig
import math
import numpy as np

def test_det():
    a = [[1.0, 2.0], [3.0, 4.0]]
    d = det(a)
    assert math.isclose(d, -2.0, abs_tol=1e-5)

def test_inv():
    a = [[1.0, 2.0], [3.0, 4.0]]
    i = inv(a)
    assert math.isclose(i[0][0], -2.0, abs_tol=1e-5)
    assert math.isclose(i[0][1], 1.0, abs_tol=1e-5)
    assert math.isclose(i[1][0], 1.5, abs_tol=1e-5)
    assert math.isclose(i[1][1], -0.5, abs_tol=1e-5)

def test_matrix_trace():
    a = [[1.0, 2.0], [3.0, 4.0]]
    assert math.isclose(matrix_trace(a), 5.0)

def test_qr():
    a = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
    q, r = qr(a)
    assert len(q) == 3
    assert len(r) == 3

def test_svd():
    a = [[1.0, 0.0], [0.0, 1.0]]
    u, s, vt = svd(a)
    assert len(s) == 2
    assert math.isclose(s[0], 1.0)
    assert math.isclose(s[1], 1.0)

def test_eig():
    a = [[2.0, 0.0], [0.0, 3.0]]
    vals, vecs = eig(a)
    assert set(vals) == {2.0, 3.0}
    assert len(vecs) == 2
