class ExecutionContext:
    """Immutable container that carries per-call execution parameters."""
    __slots__ = ("_what_lense_enabled", "_how_lense_enabled", "_trace_enabled")

    def __init__(self, trace_enabled=False, what_lense_enabled=True, how_lense_enabled=False):
        object.__setattr__(self, "_trace_enabled", trace_enabled)
        object.__setattr__(self, "_what_lense_enabled", what_lense_enabled)
        object.__setattr__(self, "_how_lense_enabled", how_lense_enabled)

    @property
    def trace_enabled(self) -> bool:
        return self._trace_enabled

    @property
    def what_lense_enabled(self) -> bool:
        return self._what_lense_enabled

    @property
    def how_lense_enabled(self) -> bool:
        return self._how_lense_enabled
