# ==============================
# File: linalg/tests/diagnostics/test_condition_number.py
# ==============================
"""Tests for condition number calculation."""

from mllense.math.linalg.diagnostics.condition_number import condition_number
import math

def test_condition_number_identity():
    a = [[1.0, 0.0], [0.0, 1.0]]
    c = condition_number(a)
    assert math.isclose(c, 1.0)

def test_condition_number_singular():
    a = [[1.0, 2.0], [2.0, 4.0]]
    c = condition_number(a)
    assert c == float("inf")
