# ==============================
# File: linalg/nn/gradients.py
# ==============================
"""Numerical gradient computation for testing and educational purposes."""

from __future__ import annotations

import math
from typing import Callable, List

from mllense.math.linalg.core.types import InternalVector

__all__ = ["numerical_gradient"]


def numerical_gradient(
    f: Callable[[InternalVector], float],
    x: InternalVector,
    *,
    epsilon: float = 1e-7,
) -> InternalVector:
    """Compute the numerical gradient of ``f`` at point ``x`` using central differences.

    ``grad_i ≈ (f(x + ε * e_i) - f(x - ε * e_i)) / (2ε)``

    Args:
        f: A scalar-valued function of a vector.
        x: The point at which to evaluate the gradient.
        epsilon: Step size for finite differences.

    Returns:
        Gradient vector of same length as ``x``.
    """
    n = len(x)
    grad: InternalVector = [0.0] * n

    for i in range(n):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += epsilon
        x_minus[i] -= epsilon

        f_plus = f(x_plus)
        f_minus = f(x_minus)

        grad[i] = (f_plus - f_minus) / (2.0 * epsilon)

    return grad
