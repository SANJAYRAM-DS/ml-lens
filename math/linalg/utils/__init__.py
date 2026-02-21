# ==============================
# File: linalg/utils/__init__.py
# ==============================
"""Utility helpers for the linalg engine."""

from mllense.math.linalg.utils.block_ops import (
    split_into_blocks,
    merge_blocks,
)
from mllense.math.linalg.utils.inspection import (
    describe_matrix,
    is_symmetric,
    is_upper_triangular,
    is_lower_triangular,
    is_diagonal,
    is_identity,
)
from mllense.math.linalg.utils.logging import get_logger
from mllense.math.linalg.utils.matrix_helpers import (
    copy_matrix,
    copy_vector,
    add_matrices,
    subtract_matrices,
    scale_matrix,
    flatten_matrix,
)
from mllense.math.linalg.utils.performance import benchmark, BenchmarkResult

__all__ = [
    "split_into_blocks",
    "merge_blocks",
    "describe_matrix",
    "is_symmetric",
    "is_upper_triangular",
    "is_lower_triangular",
    "is_diagonal",
    "is_identity",
    "get_logger",
    "copy_matrix",
    "copy_vector",
    "add_matrices",
    "subtract_matrices",
    "scale_matrix",
    "flatten_matrix",
    "benchmark",
    "BenchmarkResult",
]
