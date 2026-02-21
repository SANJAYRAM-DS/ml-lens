# ==============================
# File: linalg/algorithms/base.py
# ==============================
"""Abstract base class for all algorithms.

Every algorithm must:
1.  Inherit from :class:`BaseAlgorithm`.
2.  Declare a :attr:`metadata` class attribute.
3.  Implement :meth:`execute`.
"""

from __future__ import annotations

import abc
from typing import Any

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.trace import Trace

__all__ = ["BaseAlgorithm"]


class BaseAlgorithm(abc.ABC):
    """Base class for pluggable algorithms.

    Subclasses must set the ``metadata`` class attribute and implement
    :meth:`execute`.

    The ``trace`` parameter is *always* passed in by the API layer.
    Algorithms should call ``trace.record(...)`` at interesting points
    when tracing is enabled.
    """

    metadata: AlgorithmMetadata

    @abc.abstractmethod
    def execute(
        self,
        *args: Any,
        context: ExecutionContext,
        trace: Trace,
        **kwargs: Any,
    ) -> Any:
        """Run the algorithm.

        Args:
            *args: Positional operands (matrices / vectors).
            context: The immutable execution context.
            trace: A :class:`Trace` instance (may be disabled).
            **kwargs: Algorithm-specific keyword arguments.

        Returns:
            The computation result in *internal* format.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(operation={self.metadata.operation!r})"