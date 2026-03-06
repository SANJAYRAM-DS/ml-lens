# ==============================
# File: linalg/backend/__init__.py
# ==============================
"""Backend subsystem — swappable compute engines."""

from mllense.math.linalg.backend.base import Backend
from mllense.math.linalg.backend.numpy_backend import NumpyBackend
from mllense.math.linalg.backend.python_backend import PythonBackend

__all__ = ["Backend", "NumpyBackend", "PythonBackend"]
