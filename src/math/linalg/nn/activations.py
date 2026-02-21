# ==============================
# File: linalg/nn/activations.py
# ==============================
"""Activation functions built on linalg primitives."""

from __future__ import annotations

import math
from typing import List, Union

from mllense.math.linalg.core.types import InternalMatrix, InternalVector

__all__ = ["relu", "sigmoid", "tanh", "softmax"]


def relu(x: Union[InternalVector, InternalMatrix]) -> Union[InternalVector, InternalMatrix]:
    """Element-wise ReLU: ``max(0, x)``."""
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[max(0.0, v) for v in row] for row in x]
    return [max(0.0, v) for v in x]  # type: ignore[union-attr]


def sigmoid(x: Union[InternalVector, InternalMatrix]) -> Union[InternalVector, InternalMatrix]:
    """Element-wise sigmoid: ``1 / (1 + exp(-x))``."""
    def _sig(v: float) -> float:
        if v >= 0:
            return 1.0 / (1.0 + math.exp(-v))
        else:
            ev = math.exp(v)
            return ev / (1.0 + ev)

    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[_sig(v) for v in row] for row in x]
    return [_sig(v) for v in x]  # type: ignore[union-attr]


def tanh(x: Union[InternalVector, InternalMatrix]) -> Union[InternalVector, InternalMatrix]:
    """Element-wise tanh."""
    if isinstance(x, list) and x and isinstance(x[0], list):
        return [[math.tanh(v) for v in row] for row in x]
    return [math.tanh(v) for v in x]  # type: ignore[union-attr]


def softmax(x: InternalVector) -> InternalVector:
    """Softmax over a 1-D vector (numerically stable).

    ``softmax(x_i) = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))``
    """
    max_val = max(x)
    exps = [math.exp(v - max_val) for v in x]
    total = math.fsum(exps)
    return [e / total for e in exps]
