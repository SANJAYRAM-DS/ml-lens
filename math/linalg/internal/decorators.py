# ==============================
# File: linalg/internal/decorators.py
# ==============================
"""Re-export of _internal decorators for backward-compatibility."""

from mllense.math.linalg._internal.decorators import (
    algorithm,
    register_backend,
    timed,
    validate_inputs,
)

__all__ = ["algorithm", "register_backend", "timed", "validate_inputs"]
