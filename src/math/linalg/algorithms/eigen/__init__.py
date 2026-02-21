# ==============================
# File: linalg/algorithms/eigen/__init__.py
# ==============================
"""Eigen algorithm family."""

from mllense.math.linalg.algorithms.eigen.dominant import DominantEigen
from mllense.math.linalg.algorithms.eigen.power_iteration import PowerIteration

__all__ = ["PowerIteration", "DominantEigen"]
