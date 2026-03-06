# ==============================
# File: linalg/algorithms/shape/base.py
# ==============================
"""Base class for shape-manipulation algorithm family."""

from __future__ import annotations

from mllense.math.linalg.algorithms.base import BaseAlgorithm

__all__ = ["BaseShape"]


class BaseShape(BaseAlgorithm):
    """Abstract base for shape-manipulation algorithms."""
