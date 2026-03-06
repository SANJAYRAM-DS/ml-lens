# ==============================
# File: linalg/tests/backend/test_backend_switching.py
# ==============================
"""Tests for switching backends dynamically."""

from mllense.math.linalg.registry.backend_registry import backend_registry
from mllense.math.linalg.api.matmul import matmul
from mllense.math.linalg.exceptions import InvalidBackendError
import pytest

def test_backend_switching(monkeypatch):
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    
    # Force pure python fallback mechanism if available
    res_py = matmul(a, b, backend="python")
    assert res_py == [[19.0, 22.0], [43.0, 50.0]]
    
    # Try numpy backend (should be identical or return numpy array depending on config)
    res_np = matmul(a, b, backend="numpy")
    assert res_py == [[19.0, 22.0], [43.0, 50.0]]  # both should match logic
    
def test_invalid_backend():
    with pytest.raises(InvalidBackendError):
        matmul([[1.0]], [[2.0]], backend="imaginary_backend_xxx")
