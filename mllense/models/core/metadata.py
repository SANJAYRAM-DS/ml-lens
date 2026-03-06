from dataclasses import dataclass

@dataclass(frozen=True)
class ModelMetadata:
    name: str
    model_type: str
    complexity: str
    description: str

class ModelResult:
    """Wrapper for model prediction operations to support educational observability."""
    def __init__(self, value, what_lense="", how_lense="", metadata=None):
        self.value = value
        self.what_lense = what_lense
        self.how_lense = how_lense
        self.metadata = metadata

    @property
    def algorithm_used(self):
        return self.metadata.name if self.metadata else "Unknown"

    def __repr__(self):
        return repr(self.value)
