# ==============================
# File: linalg/registry/__init__.py
# ==============================
"""Registry subsystem — connects algorithms and backends."""

from mllense.math.linalg.registry.algorithm_registry import (
    AlgorithmRegistry,
    algorithm_registry,
)
from mllense.math.linalg.registry.auto_register import auto_register
from mllense.math.linalg.registry.backend_registry import (
    BackendRegistry,
    backend_registry,
)

__all__ = [
    "AlgorithmRegistry",
    "algorithm_registry",
    "BackendRegistry",
    "backend_registry",
    "auto_register",
]
