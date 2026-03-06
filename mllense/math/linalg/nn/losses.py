# ==============================
# File: linalg/nn/losses.py
# ==============================
"""Loss functions built on linalg primitives."""

from __future__ import annotations

import math
from typing import List

from mllense.math.linalg.core.types import InternalVector
from mllense.math.linalg.exceptions import ShapeMismatchError

__all__ = ["mse_loss", "cross_entropy_loss"]


def mse_loss(predicted: InternalVector, target: InternalVector) -> float:
    """Mean Squared Error loss.

    ``MSE = (1/n) Σ (predicted_i - target_i)^2``
    """
    if len(predicted) != len(target):
        raise ShapeMismatchError(
            expected=f"same length ({len(target)})",
            got=f"length {len(predicted)}",
            operation="mse_loss",
        )
    n = len(predicted)
    if n == 0:
        return 0.0
    return math.fsum((predicted[i] - target[i]) ** 2 for i in range(n)) / n


def cross_entropy_loss(predicted: InternalVector, target: InternalVector) -> float:
    """Binary cross-entropy loss.

    ``CE = -(1/n) Σ [target_i * log(p_i) + (1 - target_i) * log(1 - p_i)]``

    Predicted values are clipped to ``[eps, 1 - eps]`` for numerical stability.
    """
    if len(predicted) != len(target):
        raise ShapeMismatchError(
            expected=f"same length ({len(target)})",
            got=f"length {len(predicted)}",
            operation="cross_entropy_loss",
        )
    n = len(predicted)
    if n == 0:
        return 0.0

    eps = 1e-15
    total = 0.0
    for i in range(n):
        p = max(eps, min(1.0 - eps, predicted[i]))
        t = target[i]
        total += t * math.log(p) + (1.0 - t) * math.log(1.0 - p)
    return -total / n
