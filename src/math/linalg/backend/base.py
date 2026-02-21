# ==============================
# File: linalg/backend/base.py
# ==============================
"""Abstract base class for all compute backends.

Every backend must implement this interface.  The API layer never
calls backend methods directly — algorithms do.
"""

from __future__ import annotations

import abc
from typing import List

from mllense.math.linalg.core.types import InternalMatrix, InternalVector

__all__ = ["Backend"]


class Backend(abc.ABC):
    """Unified compute backend interface.

    All operations accept and return *internal* types
    (``list[list[float]]`` / ``list[float]``).
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Return the canonical name of this backend (e.g. ``"numpy"``)."""

    # ── matrix operations ────────────────────────────────────────────── #

    @abc.abstractmethod
    def matmul(self, a: InternalMatrix, b: InternalMatrix) -> InternalMatrix:
        """Matrix multiplication ``A @ B``."""

    @abc.abstractmethod
    def dot(self, a: InternalVector, b: InternalVector) -> float:
        """Dot product of two vectors."""

    @abc.abstractmethod
    def solve(self, a: InternalMatrix, b: InternalVector) -> InternalVector:
        """Solve ``A x = b`` for ``x``."""

    @abc.abstractmethod
    def inverse(self, a: InternalMatrix) -> InternalMatrix:
        """Compute the inverse of square matrix ``A``."""

    @abc.abstractmethod
    def transpose(self, a: InternalMatrix) -> InternalMatrix:
        """Return the transpose of ``A``."""

    @abc.abstractmethod
    def zeros(self, rows: int, cols: int) -> InternalMatrix:
        """Return a zero matrix of shape ``(rows, cols)``."""

    @abc.abstractmethod
    def identity(self, n: int) -> InternalMatrix:
        """Return the ``n × n`` identity matrix."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
