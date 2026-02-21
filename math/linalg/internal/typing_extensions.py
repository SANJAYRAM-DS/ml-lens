# ==============================
# File: linalg/internal/typing_extensions.py
# ==============================
"""Re-export of _internal typing extensions for backward-compatibility."""

from mllense.math.linalg._internal.typing_extensions import (
    AlgorithmMap,
    BackendMap,
    ColumnVector,
    ElementFactory,
    Number,
    RowVector,
    Shape1D,
    Shape2D,
    ShapeLike,
    SquareMatrix,
    VectorFactory,
)

__all__ = [
    "Number",
    "RowVector",
    "ColumnVector",
    "SquareMatrix",
    "ElementFactory",
    "VectorFactory",
    "Shape2D",
    "Shape1D",
    "ShapeLike",
    "AlgorithmMap",
    "BackendMap",
]
