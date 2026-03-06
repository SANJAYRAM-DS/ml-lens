# ==============================
# File: linalg/core/trace.py
# ==============================
"""Step-by-step operation tracer for educational and debug modes.

Trace recording is **off by default** and controlled per-call via
:attr:`ExecutionContext.trace_enabled`.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, List

__all__ = ["TraceStep", "Trace"]


@dataclass
class TraceStep:
    """One recorded step of a computation.

    Attributes:
        step_number: 1-based ordinal within the current trace.
        operation: Human-readable operation name (e.g. ``"pivot_swap"``).
        description: Prose explanation of what happened.
        data: Snapshot of the matrix / vector at this point (deep-copied).
        complexity_note: Optional note about the cost.
    """

    step_number: int
    operation: str
    description: str = ""
    data: Any = None
    complexity_note: str = ""


class Trace:
    """Accumulates :class:`TraceStep` instances during a computation.

    Create one ``Trace`` per API call.  Pass it into algorithms that
    support recording.  After the call, inspect :attr:`steps` or call
    :meth:`replay` for a human-readable log.
    """

    def __init__(self, enabled: bool = False) -> None:
        self._enabled: bool = enabled
        self._steps: List[TraceStep] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def steps(self) -> List[TraceStep]:
        return list(self._steps)

    def record(
        self,
        operation: str,
        description: str = "",
        data: Any = None,
        complexity_note: str = "",
    ) -> None:
        """Record a step if tracing is enabled.

        The *data* argument is **deep-copied** so that later mutations
        do not corrupt the trace history.
        """
        if not self._enabled:
            return
        step = TraceStep(
            step_number=len(self._steps) + 1,
            operation=operation,
            description=description,
            data=copy.deepcopy(data) if data is not None else None,
            complexity_note=complexity_note,
        )
        self._steps.append(step)

    def _format_matrix(self, matrix: list[list[float]] | Any) -> str:
        """Helper to format a matrix with truncation for large matrices."""
        if not isinstance(matrix, list) or not matrix or not isinstance(matrix[0], list):
            return str(matrix)

        rows = len(matrix)
        cols = len(matrix[0]) if rows > 0 else 0
        if rows <= 10 and cols <= 10:
            return str(matrix)

        lines = ["["]
        for i in range(rows):
            if rows > 10 and 5 <= i < rows - 5:
                if i == 5:
                    lines.append("  [ ... intermediate rows omitted ... ],")
                continue
                
            row_str = "  ["
            for j in range(cols):
                if cols > 10 and 5 <= j < cols - 5:
                    if j == 5:
                        row_str += " ..., "
                    continue
                row_str += f"{matrix[i][j]:.4g}, "
            row_str = row_str.rstrip(", ") + "],"
            lines.append(row_str)
        lines.append("]")
        return "\n".join(lines)

    def replay(self) -> str:
        """Return a formatted multi-line string of all recorded steps."""
        if not self._steps:
            return "(no trace recorded)"
        lines: list[str] = []
        for s in self._steps:
            header = f"Step {s.step_number}: [{s.operation}]"
            if s.description:
                header += f" — {s.description}"
            lines.append(header)
            if s.data is not None:
                formatted_data = self._format_matrix(s.data)
                lines.append(f"  Data: {formatted_data}")
            if s.complexity_note:
                lines.append(f"  Complexity: {s.complexity_note}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Discard all recorded steps."""
        self._steps.clear()

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        return f"Trace(enabled={self._enabled}, steps={len(self._steps)})"
