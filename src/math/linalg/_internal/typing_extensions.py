# ==============================
# File: linalg/_internal/typing_extensions.py
# ==============================
"""Extended type aliases used internally across the linalg engine.

These supplement the public types in ``core.types`` with more granular
internal-only aliases.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# ── Scalar types ─────────────────────────────────────────────────────────── #
Number = Union[int, float, np.integer, np.floating]
"""Any numeric scalar the engine accepts."""

# ── Matrix / vector types ────────────────────────────────────────────────── #
RowVector = List[float]
ColumnVector = List[List[float]]  # [[x0], [x1], ...]
SquareMatrix = List[List[float]]

# ── Callback / factory types ─────────────────────────────────────────────── #
ElementFactory = Callable[[int, int], float]
"""A callable (row, col) → float used by creation algorithms."""

VectorFactory = Callable[[int], float]
"""A callable (index) → float used by vector creation algorithms."""

# ── Shape types ──────────────────────────────────────────────────────────── #
Shape2D = Tuple[int, int]
Shape1D = Tuple[int]
ShapeLike = Union[Shape1D, Shape2D, List[int], Tuple[int, ...]]

# ── Registry types ───────────────────────────────────────────────────────── #
AlgorithmMap = Dict[str, Dict[str, Any]]
BackendMap = Dict[str, Any]
