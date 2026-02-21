# ==============================
# File: linalg/algorithms/decomposition/base.py
# ==============================
"""Base class for decomposition algorithm family."""

from __future__ import annotations

from mllense.math.linalg.algorithms.base import BaseAlgorithm

__all__ = ["BaseDecomposition"]


class BaseDecomposition(BaseAlgorithm):
    """Abstract base for matrix decomposition algorithms."""
