import abc
from mllense.models.core.trace import Trace
from mllense.models.core.execution_context import ExecutionContext

class BaseEstimator(abc.ABC):
    """
    Base class for all models in mllense.models mimicking sklearn
    and adding what/how lenses similar to math/linalg layout.
    """
    metadata = None

    def __init__(self, what_lense=False, how_lense=False):
        self.what_lense_enabled = what_lense
        self.how_lense_enabled = how_lense
        self.context = ExecutionContext(
            trace_enabled=how_lense,
            what_lense_enabled=what_lense,
            how_lense_enabled=how_lense
        )
        self.what_lense = ""
        self.how_lense = ""

    def _generate_what_lense(self, context=None) -> str:
        if getattr(self, "metadata", None):
            lines = [
                f"=== WHAT: {self.metadata.name} ===",
                self.metadata.description,
                "",
                "=== WHY it exists ===",
                f"Core algorithm for {self.metadata.model_type}.",
                ""
            ]
            return "\n".join(lines)
        return "No explanation available."

    def _finalize_how_lense(self, trace: Trace) -> str:
        if not trace.enabled:
            return ""
        steps = []
        for step in trace.steps:
            steps.append(f"{step.step_number}. {step.description}")
        return "\n".join(steps)
