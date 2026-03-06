# ==============================
# File: linalg/exceptions.py
# ==============================
"""Custom exceptions for the linalg engine.

All user-facing errors inherit from LinalgError.
Raw ValueError / TypeError must never be raised directly by library code.
"""

__all__ = [
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


class LinalgError(Exception):
    """Base exception for all linalg errors."""


class ShapeMismatchError(LinalgError):
    """Raised when matrix/vector shapes are incompatible for the requested operation."""

    def __init__(self, expected: str, got: str, operation: str = "") -> None:
        self.expected = expected
        self.got = got
        self.operation = operation
        msg = f"Shape mismatch: expected {expected}, got {got}"
        if operation:
            msg = f"[{operation}] {msg}"
        super().__init__(msg)


class SingularMatrixError(LinalgError):
    """Raised when an operation requires a non-singular matrix but receives a singular one."""

    def __init__(self, detail: str = "Matrix is singular or nearly singular.") -> None:
        self.detail = detail
        super().__init__(detail)


class InvalidBackendError(LinalgError):
    """Raised when a requested backend is not registered or unavailable."""

    def __init__(self, backend_name: str) -> None:
        self.backend_name = backend_name
        super().__init__(
            f"Backend '{backend_name}' is not registered. "
            f"Available backends can be listed via the backend registry."
        )


class AlgorithmNotFoundError(LinalgError):
    """Raised when a requested algorithm is not found in the algorithm registry."""

    def __init__(self, operation: str, algorithm_name: str = "") -> None:
        self.operation = operation
        self.algorithm_name = algorithm_name
        msg = f"No algorithm found for operation '{operation}'"
        if algorithm_name:
            msg += f" with name '{algorithm_name}'"
        super().__init__(msg)


class InvalidModeError(LinalgError):
    """Raised when an invalid execution mode is requested."""

    def __init__(self, mode: str, valid_modes: tuple[str, ...] = ()) -> None:
        self.mode = mode
        self.valid_modes = valid_modes
        msg = f"Invalid mode '{mode}'"
        if valid_modes:
            msg += f". Valid modes: {', '.join(valid_modes)}"
        super().__init__(msg)


class InvalidInputError(LinalgError):
    """Raised when user input is invalid (non-numeric, wrong type, etc.)."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class NonRectangularMatrixError(LinalgError):
    """Raised when a list-of-lists does not form a rectangular matrix."""

    def __init__(self, row_lengths: list[int] | None = None) -> None:
        self.row_lengths = row_lengths
        msg = "Input is not a rectangular matrix"
        if row_lengths:
            msg += f" (row lengths: {row_lengths})"
        super().__init__(msg)


class EmptyMatrixError(LinalgError):
    """Raised when an empty matrix is provided where a non-empty one is required."""

    def __init__(self, detail: str = "Empty matrix is not supported for this operation.") -> None:
        self.detail = detail
        super().__init__(detail)


class NumericalInstabilityError(LinalgError):
    """Raised when a numerical computation encounters instability (overflow, underflow, etc.)."""

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)
