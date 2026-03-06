"""Abstract base class for all algorithms.

Every algorithm must:
1.  Inherit from :class:`BaseAlgorithm`.
2.  Declare a :attr:`metadata` class attribute.
3.  Implement :meth:`execute`.
"""

from __future__ import annotations

import abc
from typing import Any, List

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

    def __init__(self) -> None:
        self._what_lense: str = ""
        self._how_lense: str = ""
        self._checkpoints: List[str] = []

    def _record_checkpoint(self, message: str) -> None:
        """Record a single step for the how_lense explanation."""
        self._checkpoints.append(message)

    def _finalize_how_lense(self) -> str:
        """Process recorded checkpoints into a truncated, readable explanation."""
        if not self._checkpoints:
            # Respect subclasses that assign self._how_lense or self.how_lense manually
            if hasattr(self, "how_lense") and self.how_lense:
                return self.how_lense
            if not self._how_lense:
                self._how_lense = ""
            return self._how_lense

        total_steps = len(self._checkpoints)
        if total_steps <= 10:
            self._how_lense = "\n".join(self._checkpoints)
        else:
            first_five = self._checkpoints[:5]
            last_five = self._checkpoints[-5:]
            self._how_lense = "\n".join(first_five + ["   ... intermediate steps omitted ..."] + last_five)
            
        # Update public facing alias for subclasses
        self.how_lense = self._how_lense
        return self._how_lense

    def _generate_what_lense(self) -> str:
        """Generate the what_lense educational explanation from metadata."""
        # Respect manually set explanations in subclasses
        if hasattr(self, "what_lense") and self.what_lense:
            return self.what_lense
        
        if getattr(self, "_what_lense", ""):
            return self._what_lense
        if not hasattr(self, "metadata") or not self.metadata:
            self._what_lense = "No explanation available."
            return self._what_lense

        lines = [
            f"=== WHAT: {self.metadata.name} ===",
            self.metadata.description,
            "",
            "=== WHY it exists ===",
            "Fundamental algorithm for its operation.",
            "",
            "=== WHERE it is used in ML / AI === ",
            "Commonly used in ML pipelines and systems.",
            ""
        ]
        self._what_lense = "\n".join(lines)
        return self._what_lense

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