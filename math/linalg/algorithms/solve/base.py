# ==============================
# File: linalg/algorithms/solve/base.py
# ==============================
"""Base class for solve algorithm family."""

from __future__ import annotations

from mllense.math.linalg.algorithms.base import BaseAlgorithm

__all__ = ["BaseSolve"]


class BaseSolve(BaseAlgorithm):
    """Abstract base for linear system solvers."""
