# ==============================
# File: linalg/algorithms/creation/base.py
# ==============================
"""Base class for matrix/vector creation algorithms."""

from __future__ import annotations

import abc
from typing import Any

from mllense.math.linalg.algorithms.base import BaseAlgorithm
from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.trace import Trace

__all__ = ["BaseCreation"]


class BaseCreation(BaseAlgorithm):
    """Abstract base for creation algorithms (zeros, ones, eye, rand)."""

    @abc.abstractmethod
    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Any:
        """Create a matrix or vector.

        Positional args vary by subclass (typically rows, cols).
        """
