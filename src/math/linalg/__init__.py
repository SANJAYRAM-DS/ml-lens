# ==============================
# File: linalg/__init__.py
# ==============================
"""linalg — Production-grade, educational, extensible linear algebra engine.

Usage::

    from mllense.math.linalg import matmul, solve

    # matrix multiplication
    C = matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])

    # solve Ax = b
    x = solve([[3, 1], [1, 2]], [9, 8])
"""

from __future__ import annotations

# ── bootstrap: register all built-in backends & algorithms ────────────── #
from mllense.math.linalg.registry.auto_register import auto_register as _auto_register

_auto_register()

# ── public API ────────────────────────────────────────────────────────── #
from mllense.math.linalg.api.matmul import matmul  # noqa: E402
from mllense.math.linalg.api.solve import solve  # noqa: E402
from mllense.math.linalg.api.create import zeros, ones, eye, rand  # noqa: E402
from mllense.math.linalg.api.ops import add, subtract, multiply, divide, scalar_add, scalar_multiply  # noqa: E402
from mllense.math.linalg.api.shape import transpose, reshape, flatten, vstack, hstack  # noqa: E402
from mllense.math.linalg.api.decomposition import det, inv, matrix_trace, qr, svd, eig  # noqa: E402
from mllense.math.linalg.api.eigen import dominant_eigen  # noqa: E402
from mllense.math.linalg.api.norms import vector_norm, frobenius_norm, spectral_norm  # noqa: E402
from mllense.math.linalg.diagnostics.condition_number import condition_number  # noqa: E402
from mllense.math.linalg.diagnostics.rank import matrix_rank  # noqa: E402
from mllense.math.linalg.diagnostics.stability import stability_report  # noqa: E402
from mllense.math.linalg.diagnostics.report import full_diagnostic_report  # noqa: E402

from mllense.math.linalg.config import GlobalConfig, get_config  # noqa: E402
from mllense.math.linalg.version import __version__  # noqa: E402

# ── re-export exceptions for convenience ─────────────────────────────── #
from mllense.math.linalg.exceptions import (  # noqa: E402
    AlgorithmNotFoundError,
    EmptyMatrixError,
    InvalidBackendError,
    InvalidInputError,
    InvalidModeError,
    LinalgError,
    NonRectangularMatrixError,
    NumericalInstabilityError,
    ShapeMismatchError,
    SingularMatrixError,
)

__all__ = [
    # API
    "matmul",
    "solve",
    "zeros",
    "ones",
    "eye",
    "rand",
    "add",
    "subtract",
    "multiply",
    "divide",
    "scalar_add",
    "scalar_multiply",
    "transpose",
    "reshape",
    "flatten",
    "vstack",
    "hstack",
    "det",
    "inv",
    "matrix_trace",
    "qr",
    "svd",
    "eig",
    "dominant_eigen",
    "vector_norm",
    "frobenius_norm",
    "spectral_norm",
    "condition_number",
    "matrix_rank",
    "stability_report",
    "full_diagnostic_report",
    # Configuration
    "GlobalConfig",
    "get_config",
    # Version
    "__version__",
    # Exceptions
    "LinalgError",
    "ShapeMismatchError",
    "SingularMatrixError",
    "InvalidBackendError",
    "AlgorithmNotFoundError",
    "InvalidModeError",
    "InvalidInputError",
    "NonRectangularMatrixError",
    "EmptyMatrixError",
    "NumericalInstabilityError",
]
