# ==============================
# File: linalg/core/metadata.py
# ==============================
"""Algorithm metadata descriptor.

Each algorithm declares a :class:`AlgorithmMetadata` instance that the
registry uses for auto-selection and documentation purposes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["AlgorithmMetadata"]


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
