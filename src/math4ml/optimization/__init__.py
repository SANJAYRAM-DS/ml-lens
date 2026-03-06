"""
===============================================
Optimization Module (math4ml.optimization)
===============================================

This module contains derivative tools and optimizers used across
machine learning training pipelines.

SUBMODULES
----------

1. gradients
    Tools for computing numerical derivatives:
    - grad(f, x, help=False, what=False)
    - jacobian(f, x, help=False, what=False)
    - hessian(f, x, help=False, what=False)

    These help understand sensitivity, curvature, and optimization behavior.

2. optimizers
    Core gradient-based optimization algorithms:
    - gradient_descent(f, x0, lr, epochs)
    - momentum_optimizer(grad_f, x0, lr, beta)
    - adam_optimizer(grad_f, x0)

HELP & WHAT SYSTEM
------------------
All functions accept two optional flags:

help=True  
    Returns step-by-step derivation/updates.

what=True  
    Returns a conceptual explanation + example.

Example:
--------
x, help_text, what_text = gradient_descent(f, [1,2], help=True, what=True)

print(help_text)
print(what_text)

USAGE
-----

import math4ml.optimization as opt

g = opt.gradients.grad(f, x)
x_new = opt.optimizers.gradient_descent(f, x0)

To explore submodules:
help(math4ml.optimization.gradients)
help(math4ml.optimization.optimizers)
"""

from . import gradients
from . import optimizers

__all__ = ["gradients", "optimizers"]
