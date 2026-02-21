# ==============================
# File: linalg/tests/diagnostics/test_rank.py
# ==============================
"""Tests for matrix rank estimation."""

from mllense.math.linalg.diagnostics.rank import matrix_rank

def test_rank_full():
    a = [[1.0, 0.0], [0.0, 1.0]]
    assert matrix_rank(a) == 2
    
def test_rank_deficient():
    a = [[1.0, 2.0], [2.0, 4.0]]
    assert matrix_rank(a) == 1

def test_rank_zero():
    a = [[0.0, 0.0], [0.0, 0.0]]
    assert matrix_rank(a) == 0
