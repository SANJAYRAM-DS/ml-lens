import copy
from dataclasses import dataclass
from typing import Any, List

@dataclass
class TraceStep:
    step_number: int
    operation: str
    description: str = ""
    data: Any = None
    complexity_note: str = ""

class Trace:
    def __init__(self, enabled: bool = False):
        self._enabled = enabled
        self._steps: List[TraceStep] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def steps(self) -> List[TraceStep]:
        return list(self._steps)

    def record(self, operation: str, description: str = "", data: Any = None, complexity_note: str = "") -> None:
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

    def clear(self) -> None:
        self._steps.clear()
        
    def replay(self) -> str:
        if not self._steps:
            return "(no trace recorded)"
        lines = []
        for s in self._steps:
            header = f"Step {s.step_number}: [{s.operation}]"
            if s.description:
                header += f" — {s.description}"
            lines.append(header)
        return "\n".join(lines)
