# ==============================
# File: linalg/algorithms/norms/base.py
# ==============================
"""Base class for norm algorithm family."""

from __future__ import annotations

from mllense.math.linalg.algorithms.base import BaseAlgorithm

__all__ = ["BaseNorm"]


class BaseNorm(BaseAlgorithm):
    """Abstract base for norm algorithms."""
