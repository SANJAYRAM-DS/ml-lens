# ==============================
# File: linalg/core/complexity.py
# ==============================
"""Computational complexity estimation utilities.

Used by the trace and educational-mode systems to annotate
algorithm steps with theoretical and actual operation counts.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ComplexityInfo", "estimate_matmul_complexity", "estimate_solve_complexity"]


@dataclass(frozen=True)
class ComplexityInfo:
    """Container for complexity estimates.

    Attributes:
        theoretical: Big-O expression as a string.
        estimated_flops: Estimated number of floating-point operations.
        estimated_memory_bytes: Estimated memory footprint in bytes.
    """

    theoretical: str
    estimated_flops: int
    estimated_memory_bytes: int

    def __repr__(self) -> str:
        return (
            f"ComplexityInfo(theoretical={self.theoretical!r}, "
            f"flops={self.estimated_flops:,}, "
            f"memory={self.estimated_memory_bytes:,} bytes)"
        )


def estimate_matmul_complexity(m: int, k: int, n: int) -> ComplexityInfo:
    """Estimate complexity for naive ``(m×k) @ (k×n)`` multiplication.

    * FLOPs: ``2 * m * k * n``  (one multiply + one add per element contribution).
    * Memory: ``8 * (m*k + k*n + m*n)``  (64-bit floats for A, B, and result).
    """
    flops = 2 * m * k * n
    mem = 8 * (m * k + k * n + m * n)
    return ComplexityInfo(
        theoretical=f"O({m}*{k}*{n})",
        estimated_flops=flops,
        estimated_memory_bytes=mem,
    )


def estimate_solve_complexity(n: int) -> ComplexityInfo:
    """Estimate complexity for Gaussian elimination on an ``n×n`` system.

    * FLOPs: roughly ``(2/3) * n³``.
    * Memory: ``8 * n * (n + 1)``  (augmented matrix).
    """
    flops = int((2.0 / 3.0) * n ** 3)
    mem = 8 * n * (n + 1)
    return ComplexityInfo(
        theoretical=f"O(n^3) where n={n}",
        estimated_flops=flops,
        estimated_memory_bytes=mem,
    )


def estimate_elementwise_complexity(rows: int, cols: int) -> ComplexityInfo:
    """Estimate complexity for element-wise operations."""
    flops = rows * cols
    mem = 8 * rows * cols * 2  # input + output
    return ComplexityInfo(
        theoretical=f"O({rows}*{cols})",
        estimated_flops=flops,
        estimated_memory_bytes=mem,
    )


def estimate_transpose_complexity(rows: int, cols: int) -> ComplexityInfo:
    """Estimate complexity for matrix transpose."""
    flops = rows * cols  # each element copied once
    mem = 8 * rows * cols * 2
    return ComplexityInfo(
        theoretical=f"O({rows}*{cols})",
        estimated_flops=flops,
        estimated_memory_bytes=mem,
    )


def estimate_decomposition_complexity(n: int, algorithm: str = "generic") -> ComplexityInfo:
    """Estimate complexity for matrix decomposition algorithms."""
    if algorithm == "lu":
        flops = int((2.0 / 3.0) * n ** 3)
    elif algorithm == "qr":
        flops = int((4.0 / 3.0) * n ** 3)
    elif algorithm == "svd":
        flops = int(4 * n ** 3)
    elif algorithm == "cholesky":
        flops = int((1.0 / 3.0) * n ** 3)
    else:
        flops = int(n ** 3)
    mem = 8 * n * n * 3  # multiple matrices in play
    return ComplexityInfo(
        theoretical=f"O(n^3) where n={n}, alg={algorithm}",
        estimated_flops=flops,
        estimated_memory_bytes=mem,
    )
