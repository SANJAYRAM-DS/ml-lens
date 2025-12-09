"""
===============================================
Linear Algebra Module (math4ml.linalg)
===============================================

This module provides core linear algebra tools used
in machine learning and scientific computing.

SUBMODULES
----------

1. create
    Functions for creating matrices and vectors.
    - zeros(shape)
    - ones(shape)
    - eye(n)
    - rand(shape, seed=None)

2. ops
    Elementwise operations.
    - add(A, B)
    - subtract(A, B)
    - multiply(A, B)
    - divide(A, B)
    - scalar_multiply(A, s)

3. shape
    Reshaping and dimensional helpers.
    - reshape(A, new_shape)
    - flatten(A)
    - stack(A, B, axis)
    - concat(list_of_arrays, axis)

4. matmul
    - transpose(A)
    - matmul(A, B)
    - dot(a, b)
    - outer(a, b)

5. decomposition
    - trace(A)
    - det(A)
    - rank(A)
    - inverse(A)

6. eigen
    - power_iteration(A, num_iter, tol)
    - eigenvalues(A)
    - eigenvectors(A)

7. opt
    Optimization helpers for linalg.
    - grad_matmul(dL_dC, A, B)
    - grad_relu(x)
    - sigmoid(x)
    - grad_sigmoid(x)
    - softmax(x)
    - cross_entropy(pred, target)

8. solve
    Solving linear systems.
    - gaussian_elimination(A, b)
    - back_substitution(U, y)
    - solve(A, b)

Every function here will have excess two parameters with default parameter of False => help and what
help -> give the step by step working of the function
what -> give a basic defination with example of a particular function

Example:
A = [[2, 1], [4, 3]]
b = [5, 11]

x, help_text, what_text = solve(A, b, help=True, what=True)

print("Answer:", x)
print("\nHELP:\n", help_text)
print("\nWHAT:\n", what_text)

USAGE
-----

import math4ml.linalg as la

# creation
A = la.create.zeros((3, 3))
B = la.create.ones((3, 3))

# operations
C = la.ops.add(A, B)

# matrix multiplication
D = la.matmul.matmul(A, B)

# decomposition
vals = la.eigen.eigenvalues(A)

To explore a submodule:
help(math4ml.linalg.create)
help(math4ml.linalg.ops)
help(math4ml.linalg.matmul)

"""

# Expose submodules to the namespace
from . import create
from . import ops
from . import shape
from . import matmul
from . import decomposition
from . import eigen
from . import opt
from . import solve

__all__ = [
    "create",
    "ops",
    "shape",
    "matmul",
    "decomposition",
    "eigen",
    "opt",
    "solve",
]
