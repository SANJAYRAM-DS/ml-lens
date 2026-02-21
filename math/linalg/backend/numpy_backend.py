# ==============================
# File: linalg/backend/numpy_backend.py
# ==============================
"""NumPy-accelerated backend.

Delegates to ``numpy.matmul``, ``numpy.linalg.solve``, etc. and converts
between internal ``list[list[float]]`` and ``ndarray`` at the boundary.
"""

from __future__ import annotations

import numpy as np

from mllense.math.linalg.backend.base import Backend
from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.exceptions import SingularMatrixError

__all__ = ["NumpyBackend"]


class NumpyBackend(Backend):
    """Backend that delegates heavy lifting to NumPy."""

    @property
    def name(self) -> str:
        return "numpy"

    # ── internal conversion helpers ──────────────────────────────────── #

    @staticmethod
    def _to_np(m: InternalMatrix) -> np.ndarray:
        return np.array(m, dtype=np.float64)

    @staticmethod
    def _from_np_matrix(arr: np.ndarray) -> InternalMatrix:
        return arr.tolist()

    @staticmethod
    def _from_np_vector(arr: np.ndarray) -> InternalVector:
        return arr.tolist()

    # ── matmul ────────────────────────────────────────────────────────── #

    def matmul(self, a: InternalMatrix, b: InternalMatrix) -> InternalMatrix:
        a_np = self._to_np(a)
        b_np = self._to_np(b)
        result = np.matmul(a_np, b_np)
        if result.ndim == 0:
            return [[float(result)]]
        if result.ndim == 1:
            return [result.tolist()]
        return self._from_np_matrix(result)

    # ── dot ───────────────────────────────────────────────────────────── #

    def dot(self, a: InternalVector, b: InternalVector) -> float:
        return float(np.dot(np.array(a, dtype=np.float64),
                            np.array(b, dtype=np.float64)))

    # ── solve ─────────────────────────────────────────────────────────── #

    def solve(self, a: InternalMatrix, b: InternalVector) -> InternalVector:
        try:
            x = np.linalg.solve(
                np.array(a, dtype=np.float64),
                np.array(b, dtype=np.float64),
            )
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc
        return self._from_np_vector(x)

    # ── inverse ──────────────────────────────────────────────────────── #

    def inverse(self, a: InternalMatrix) -> InternalMatrix:
        try:
            inv = np.linalg.inv(np.array(a, dtype=np.float64))
        except np.linalg.LinAlgError as exc:
            raise SingularMatrixError(str(exc)) from exc
        return self._from_np_matrix(inv)

    # ── transpose ─────────────────────────────────────────────────────── #

    def transpose(self, a: InternalMatrix) -> InternalMatrix:
        return self._from_np_matrix(np.array(a, dtype=np.float64).T)

    # ── factory helpers ──────────────────────────────────────────────── #

    def zeros(self, rows: int, cols: int) -> InternalMatrix:
        return self._from_np_matrix(np.zeros((rows, cols), dtype=np.float64))

    def identity(self, n: int) -> InternalMatrix:
        return self._from_np_matrix(np.eye(n, dtype=np.float64))
