# ==============================
# File: linalg/algorithms/matmul/base.py
# ==============================
"""Base class for matmul algorithm family."""

from __future__ import annotations

import abc
from typing import Any

from mllense.math.linalg.algorithms.base import BaseAlgorithm

__all__ = ["BaseMatmul"]


class BaseMatmul(BaseAlgorithm):
    """Abstract base for matrix multiplication algorithms."""
