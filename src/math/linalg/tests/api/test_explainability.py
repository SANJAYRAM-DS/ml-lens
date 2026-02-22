import pytest
import numpy as np
from mllense.math.linalg import eye, matmul, rand
from mllense.math.linalg.core.metadata import LinalgResult

def test_explainability_defaults():
    # what_lense default True, how_lense default False
    result = eye(3)
    assert isinstance(result, LinalgResult)
    assert result.what_lense != ""
    assert result.how_lense == ""

def test_explainability_how_enabled():
    result = eye(3, how_lense=True)
    assert result.what_lense != ""
    assert result.how_lense != ""
    assert "1. Validated inputs" in result.how_lense

def test_correctness_unaffected():
    # Check that wrapping in LinalgResult does not break math operations
    # LinalgResult delegates to .value.
    result = eye(3)
    assert len(result) == 3
    assert result[0][0] == 1.0
    assert result[0][1] == 0.0

def test_tracing_does_not_modify_output():
    a = rand(2, 2, seed=42)
    b = rand(2, 2, seed=43)
    
    res1 = matmul(a, b, what_lense=False, how_lense=False)
    res2 = matmul(a, b, what_lense=True, how_lense=True)
    
    assert res1.value == res2.value

def test_identity_matrix_explanation():
    result = eye(2)
    what = result.what_lense.lower()
    assert "regularization" in what or "ridge regression" in what
    # Adding string checks to ensure it contains exactly required text
    # ML regularization reference
    assert "regularization" in what
    # ridge regression reference
    assert "ridge regression" in what
    # covariance stabilization could be there
    # assert "covariance stabilization" in what # let's just make sure we update eye creation or the base what_lense
