# ==============================
# File: linalg/api/__init__.py
# ==============================
"""Public API layer — thin wrappers over the registry + algorithms."""

from mllense.math.linalg.api.matmul import matmul
from mllense.math.linalg.api.solve import solve

__all__ = ["matmul", "solve"]
