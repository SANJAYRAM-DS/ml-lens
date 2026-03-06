import numpy as np

def power_iteration(A, num_iter=1000, tol=1e-9, help=False, what=False):
    A = np.array(A)
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    explanation = ""
    if help:
        explanation += "Start with a random vector and normalize it.\n"

    for it in range(num_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)

        if help:
            explanation += f"Iteration {it+1}: Multiply A*v and normalize → {v_new}\n"

        if np.linalg.norm(v_new - v) < tol:
            v = v_new
            if help:
                explanation += f"Converged after {it+1} iterations.\n"
            break

        v = v_new

    result = v.tolist()

    if help:
        explanation += f"Final dominant eigenvector approximation:\n{result}\n"

    definition = ""
    if what:
        definition += (
            "The `power_iteration` function finds the dominant eigenvector of a matrix.\n"
            "The dominant eigenvector is the direction the matrix stretches the most.\n"
            "Example: For matrix A, it repeatedly multiplies A by a vector until it stops changing.\n"
        )

    return result, explanation, definition


def eigenvalue(A, help=False, what=False):

    v, expl_vec, _ = power_iteration(A, help=True, what=False)
    λ = float(np.dot(np.array(v), np.array(A) @ np.array(v)))

    explanation = ""
    if help:
        explanation += "Step 1: Find dominant eigenvector using power iteration.\n"
        explanation += expl_vec
        explanation += f"\nStep 2: Compute eigenvalue using λ = v^T A v = {λ}\n"

    definition = ""
    if what:
        definition += (
            "The `eigenvalue` function returns the dominant eigenvalue of a matrix.\n"
            "It uses the Rayleigh quotient, λ = vᵀ A v, where v is the dominant eigenvector.\n"
            f"Example: For matrix A, eigenvalue ≈ {λ}\n"
        )

    return λ, explanation, definition


def eigenvector(A, help=False, what=False):

    v, explanation, _ = power_iteration(A, help=help, what=False)

    definition = ""
    if what:
        definition += (
            "The `eigenvector` function returns the dominant eigenvector of a matrix.\n"
            "This vector points in the direction that the matrix stretches the most.\n"
            f"Example: eigenvector(A) = {v}\n"
        )

    if not help:
        explanation = ""

    return v, explanation, definition
