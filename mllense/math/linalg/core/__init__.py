# ==============================
# File: linalg/core/__init__.py
# ==============================
"""Core infrastructure layer — dependency-safe and reusable."""

from mllense.math.linalg.core.execution_context import ExecutionContext
from mllense.math.linalg.core.metadata import AlgorithmMetadata
from mllense.math.linalg.core.mode import ExecutionMode
from mllense.math.linalg.core.trace import Trace, TraceStep
from mllense.math.linalg.core.types import (
    InternalMatrix,
    InternalVector,
    MatrixLike,
    Scalar,
    VectorLike,
    from_internal_matrix,
    from_internal_vector,
    get_matrix_shape,
    get_vector_length,
    is_numpy,
    to_internal_matrix,
    to_internal_vector,
)
from mllense.math.linalg.core.validation import (
    validate_dimension_limit,
    validate_matmul_shapes,
    validate_solve_shapes,
    validate_square,
)

__all__ = [
    "ExecutionContext",
    "ExecutionMode",
    "AlgorithmMetadata",
    "Trace",
    "TraceStep",
    "MatrixLike",
    "VectorLike",
    "Scalar",
    "InternalMatrix",
    "InternalVector",
    "to_internal_matrix",
    "to_internal_vector",
    "from_internal_matrix",
    "from_internal_vector",
    "is_numpy",
    "get_matrix_shape",
    "get_vector_length",
    "validate_matmul_shapes",
    "validate_solve_shapes",
    "validate_square",
    "validate_dimension_limit",
]
