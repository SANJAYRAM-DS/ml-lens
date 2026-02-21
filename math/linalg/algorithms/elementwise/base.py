# ==============================
# File: linalg/algorithms/elementwise/base.py
# ==============================
"""Base class for element-wise operations."""

from __future__ import annotations

import abc
from typing import Any

from mllense.math.linalg.algorithms.base import BaseAlgorithm

__all__ = ["BaseElementwise"]


class BaseElementwise(BaseAlgorithm):
    """Abstract base for element-wise matrix/vector operations."""
