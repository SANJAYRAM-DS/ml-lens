# ==============================
# File: linalg/nn/__init__.py
# ==============================
"""Neural network primitives built on top of the linalg API.

All NN code uses the linalg API layer — never directly calling numpy.
"""

from mllense.math.linalg.nn.activations import relu, sigmoid, tanh, softmax
from mllense.math.linalg.nn.losses import mse_loss, cross_entropy_loss
from mllense.math.linalg.nn.layers import linear_forward
from mllense.math.linalg.nn.gradients import numerical_gradient

__all__ = [
    "relu",
    "sigmoid",
    "tanh",
    "softmax",
    "mse_loss",
    "cross_entropy_loss",
    "linear_forward",
    "numerical_gradient",
]
