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
                lines.append(f"  Data: {s.data}")
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
