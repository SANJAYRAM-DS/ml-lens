# ==============================
# File: linalg/algorithms/eigen/base.py
# ==============================
"""Base class for eigenvalue algorithm family."""

from __future__ import annotations

from mllense.math.linalg.algorithms.base import BaseAlgorithm

__all__ = ["BaseEigen"]


class BaseEigen(BaseAlgorithm):
    """Abstract base for eigenvalue algorithms."""
