# ==============================
# File: linalg/nn/layers.py
# ==============================
"""Neural network layer primitives built on top of the linalg API.

All operations use the linalg API — never raw numpy.
"""

from __future__ import annotations

import math
from typing import Optional

from mllense.math.linalg.core.types import InternalMatrix, InternalVector
from mllense.math.linalg.exceptions import ShapeMismatchError

__all__ = ["linear_forward"]


def linear_forward(
    x: InternalMatrix,
    weights: InternalMatrix,
    bias: Optional[InternalVector] = None,
) -> InternalMatrix:
    """Compute a linear (fully-connected) forward pass: ``Y = X @ W^T + b``.

    Args:
        x: Input matrix of shape ``(batch, in_features)``.
        weights: Weight matrix of shape ``(out_features, in_features)``.
        bias: Optional bias vector of length ``out_features``.

    Returns:
        Output matrix of shape ``(batch, out_features)``.
    """
    batch = len(x)
    in_features = len(x[0]) if batch else 0
    out_features = len(weights)
    w_in = len(weights[0]) if out_features else 0

    if in_features != w_in:
        raise ShapeMismatchError(
            expected=f"x.cols ({in_features}) == weights.cols ({w_in})",
            got=f"x.cols={in_features}, weights.cols={w_in}",
            operation="linear_forward",
        )

    if bias is not None and len(bias) != out_features:
        raise ShapeMismatchError(
            expected=f"bias length == {out_features}",
            got=f"bias length == {len(bias)}",
            operation="linear_forward",
        )

    # Y = X @ W^T + b
    result: InternalMatrix = []
    for i in range(batch):
        row: list[float] = []
        for j in range(out_features):
            val = math.fsum(x[i][k] * weights[j][k] for k in range(in_features))
            if bias is not None:
                val += bias[j]
            row.append(val)
        result.append(row)

    return result
