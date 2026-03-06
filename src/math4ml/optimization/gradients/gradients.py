import numpy as np

def grad(f, x, eps=1e-6, help=False, what=False):
    """
    Compute gradient of scalar function f at x using finite differences.

    Returns:
        grad
        OR (grad, help_text, what_text)
    """
    x = np.array(x, dtype=float)
    grad_vec = np.zeros_like(x)

    explanation = []
    if help:
        explanation.append("Gradient computed using forward finite difference:\n"
                           "g[i] = (f(x+eps_i) - f(x)) / eps")

    for i in range(len(x)):
        x_eps = x.copy()
        x_eps[i] += eps
        grad_vec[i] = (f(x_eps) - f(x)) / eps

        if help:
            explanation.append(
                f"Step {i+1}: perturb x[{i}] → compute (f(x+eps) - f(x)) / eps = {grad_vec[i]}"
            )

    what_text = ("""
The gradient ∇f(x) is a vector of partial derivatives measuring 
how f changes with respect to each input dimension.

Example:
    f(x) = x₀² + x₁²
    grad(f, [1,2]) = [2*1, 2*2] = [2,4]
""") if what else None

    if help or what:
        return grad_vec, ("\n".join(explanation) if help else None), what_text
    return grad_vec


def jacobian(f, X, eps=1e-6, help=False, what=False):
    """
    Jacobian of vector-valued function f at X.

    Returns:
        J
        OR (J, help_text, what_text)
    """
    X = np.array(X, dtype=float)
    f0 = np.array(f(X))
    jac = np.zeros((len(f0), len(X)))

    explanation = []
    if help:
        explanation.append("Jacobian computed column-by-column using finite differences.")

    for i in range(len(X)):
        X_eps = X.copy()
        X_eps[i] += eps
        jac[:, i] = (np.array(f(X_eps)) - f0) / eps

        if help:
            explanation.append(f"Column {i}: df/dx[{i}] = (f(X+eps) - f(X)) / eps")

    what_text = ("""
The Jacobian J is a matrix of first-order partial derivatives for vector functions.

Example:
    f([x,y]) = [x+y, x*y]
    J = [[1, 1], [y, x]]
""") if what else None

    if help or what:
        return jac, ("\n".join(explanation) if help else None), what_text
    return jac


def hessian(f, x, eps=1e-5, help=False, what=False):
    """
    Hessian matrix using 2nd order finite differences.

    Returns:
        H
        OR (H, help_text, what_text)
    """
    x = np.array(x, dtype=float)
    n = len(x)
    hess = np.zeros((n, n))

    explanation = []
    if help:
        explanation.append("Hessian computed using second-order mixed partial finite differences.")

    for i in range(n):
        for j in range(n):
            x_ijp = x.copy(); x_ijp[i] += eps; x_ijp[j] += eps
            x_ijm = x.copy(); x_ijm[i] += eps; x_ijm[j] -= eps
            x_jim = x.copy(); x_jim[i] -= eps; x_jim[j] += eps
            x_jjm = x.copy(); x_jjm[i] -= eps; x_jjm[j] -= eps

            hess[i, j] = (f(x_ijp) - f(x_ijm) - f(x_jim) + f(x_jjm)) / (4 * eps**2)

            if help:
                explanation.append(
                    f"H[{i},{j}] = using 4-point central FD = {hess[i,j]}"
                )

    what_text = ("""
The Hessian H is a matrix of second-order derivatives. It describes curvature.

Example:
    f(x,y) = x² + 3y²
    Hessian = [[2, 0], [0, 6]]
""") if what else None

    if help or what:
        return hess, ("\n".join(explanation) if help else None), what_text
    return hess
