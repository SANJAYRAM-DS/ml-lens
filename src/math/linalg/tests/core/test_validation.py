# ==============================
# File: linalg/tests/core/test_validation.py
# ==============================
"""Tests for input validation functions."""

from mllense.math.linalg.core.validation import validate_matmul_shapes, validate_solve_shapes, validate_square
from mllense.math.linalg.exceptions import ShapeMismatchError, NonRectangularMatrixError
import pytest

def test_validate_matmul_shapes():
    m, k, n = validate_matmul_shapes((2, 3), (3, 4))
    assert m == 2
    assert k == 3
    assert n == 4
    
    with pytest.raises(ShapeMismatchError):
        validate_matmul_shapes((2, 3), (4, 5))

def test_validate_square():
    a = [[1, 0], [0, 1]]
    assert validate_square(a) == 2
    
    not_square = [[1, 0, 0], [0, 1, 0]]
    with pytest.raises(ShapeMismatchError):
        validate_square(not_square)

from mllense.math.linalg.core.types import to_internal_matrix
def test_non_rectangular():
    a = [[1, 2], [3, 4, 5]]
    with pytest.raises(NonRectangularMatrixError):
        to_internal_matrix(a)
    
def test_validate_solve():
    a = [[1, 0], [0, 1]]
    b = [2, 3]
    n = validate_solve_shapes(a, b)
    assert n == 2
    
    b_bad = [2]
    with pytest.raises(ShapeMismatchError):
        validate_solve_shapes(a, b_bad)
