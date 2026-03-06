# ==============================
# File: linalg/core/metadata.py
# ==============================
"""Algorithm metadata descriptor.

Each algorithm declares a :class:`AlgorithmMetadata` instance that the
registry uses for auto-selection and documentation purposes.
"""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass, field

__all__ = ["AlgorithmMetadata", "LinalgResult"]


@dataclass(frozen=True)
class AlgorithmMetadata:
    """Immutable descriptor that every algorithm must declare.

    Attributes:
        name: Human-readable name (e.g. ``"naive_matmul"``).
        operation: Operation family (e.g. ``"matmul"``, ``"solve"``).
        complexity: Big-O string (e.g. ``"O(n^3)"``).
        stable: Whether the algorithm is numerically stable.
        supports_batch: Whether the algorithm supports batched inputs.
        requires_square: Whether the algorithm requires square matrices.
        description: Short prose description for educational mode.
    """

    name: str
    operation: str
    complexity: str = "O(n^3)"
    stable: bool = True
    supports_batch: bool = False
    requires_square: bool = False
    description: str = ""


@dataclass
class LinalgResult:
    """Wrapper for all linear algebra algorithm operations to support educational observability.

    Attributes:
        value: The core computed mathematical value.
        what_lense: The explanation of what this algorithm does.
        how_lense: The step-by-step checkpoint record.
        metadata: The AlgorithmMetadata object associated with the operation.
        algorithm_used: The human readable name of the algorithm.
        complexity: The big-O time complexity of the algorithm.
    """
    value: Any
    what_lense: str = ""
    how_lense: str = ""
    metadata: AlgorithmMetadata | None = None
    
    @property
    def algorithm_used(self) -> str:
        return self.metadata.name if self.metadata else "Unknown"
        
    @property
    def complexity(self) -> str:
        return self.metadata.complexity if self.metadata else "Unknown"

    def __repr__(self) -> str:
        return repr(self.value)
        
    def __str__(self) -> str:
        return str(self.value)

    # To behave like the wrapped matrix value dynamically
    def __getattr__(self, name: str) -> Any:
        return getattr(self.value, name)
        
    def __len__(self) -> int:
        return len(self.value)

    def __getitem__(self, key: Any) -> Any:
        return self.value[key]

    def __iter__(self) -> Any:
        return iter(self.value)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, LinalgResult):
            return bool(self.value == other.value)
        return bool(self.value == other)
