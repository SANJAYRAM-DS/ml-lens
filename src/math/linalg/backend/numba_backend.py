# ==============================
# File: linalg/backend/numba_backend.py
# ==============================
"""Numba-accelerated backend (optional).

Falls back to the Python backend if ``numba`` is not installed.
Uses ``@njit`` for inner loops where beneficial.
"""

from __future__ import annotations

import warnings
from typing import Any

from mllense.math.linalg.backend.python_backend import PythonBackend
from mllense.math.linalg.core.types import InternalMatrix, InternalVector

__all__ = ["NumbaBackend"]

# ── Try importing numba ─────────────────────────────────────────────────── #
try:
    from numba import njit  # type: ignore[import-untyped]

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ── JIT-compiled kernels (only defined if numba available) ───────────────── #

if _HAS_NUMBA:
    import numpy as np

    @njit(cache=True)  # type: ignore[misc]
    def _numba_matmul(a: Any, b: Any) -> Any:
        m, k = a.shape
        _, n = b.shape
        result = np.zeros((m, n), dtype=np.float64)
        for i in range(m):
            for j_k in range(k):
                a_val = a[i, j_k]
                for j in range(n):
                    result[i, j] += a_val * b[j_k, j]
        return result

    @njit(cache=True)  # type: ignore[misc]
    def _numba_dot(a: Any, b: Any) -> Any:
        s = 0.0
        for i in range(a.shape[0]):
            s += a[i] * b[i]
        return s


class NumbaBackend(PythonBackend):
    """Backend that accelerates hot loops with Numba JIT.

    If Numba is not installed, silently falls back to pure-Python
    implementations inherited from :class:`PythonBackend`.
    """

    @property
    def name(self) -> str:
        return "numba"

    def matmul(self, a: InternalMatrix, b: InternalMatrix) -> InternalMatrix:
        if not _HAS_NUMBA:
            return super().matmul(a, b)
        import numpy as np

        a_np = np.array(a, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        result = _numba_matmul(a_np, b_np)
        return result.tolist()

    def dot(self, a: InternalVector, b: InternalVector) -> float:
        if not _HAS_NUMBA:
            return super().dot(a, b)
        import numpy as np

        a_np = np.array(a, dtype=np.float64)
        b_np = np.array(b, dtype=np.float64)
        return float(_numba_dot(a_np, b_np))
