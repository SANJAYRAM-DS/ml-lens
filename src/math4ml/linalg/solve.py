import numpy as np

def gaussian_elimination(A, b, help=False, what=False):


    definition = ""
    if what:
        definition += (
            "Gaussian Elimination:\n"
            "This method transforms a matrix into an upper-triangular form by eliminating\n"
            "the values below the main diagonal.\n\n"
            "Simple Example:\n"
            "A = [[2, 1], [4, 3]]\n"
            "b = [5, 11]\n"
            "After elimination, A becomes [[2, 1], [0, 1]]\n"
            "This makes the system easier to solve.\n"
        )

    A = A.astype(float)
    b = b.astype(float)
    n = len(b)
    explanation = ""

    for i in range(n):
        if abs(A[i, i]) < 1e-12:
            for j in range(i + 1, n):
                if abs(A[j, i]) > 1e-12:
                    A[[i, j]] = A[[j, i]]
                    b[[i, j]] = b[[j, i]]
                    if help:
                        explanation += (
                            f"Pivot at A[{i}][{i}] was zero.\n"
                            f"Swapped row {i} with row {j}.\n"
                            f"A now:\n{A}\nb now:\n{b}\n\n"
                        )
                    break
                
        for j in range(i + 1, n):
            if abs(A[i, i]) < 1e-12:
                raise ValueError("Matrix is singular or nearly singular.")

            factor = A[j, i] / A[i, i]
            A[j] = A[j] - factor * A[i]
            b[j] = b[j] - factor * b[i]

            if help:
                explanation += (
                    f"Eliminating A[{j}][{i}] using row {i}:\n"
                    f"Factor = {factor:.5f}\n"
                    f"New row {j}: {A[j]}\n"
                    f"New b[{j}]: {b[j]}\n\n"
                )

    if help:
        explanation += (
            "Final upper-triangular matrix A:\n"
            f"{A}\nModified b:\n{b}\n\n"
        )

    return A, b, explanation, definition



def back_substitution(U, y, help=False, what=False):


    definition = ""
    if what:
        definition += (
            "Back Substitution:\n"
            "This step solves the system from bottom to top once the matrix is upper-triangular.\n"
            "You start at the last variable and move upward.\n\n"
            "Simple Example:\n"
            "If Ux = y is:\n"
            "[[2, 1],\n"
            " [0, 3]]  * [x1, x2] = [5, 6]\n"
            "Then:\n"
            "x2 = 6 / 3 = 2\n"
            "x1 = (5 - 1*2) / 2 = 1\n"
        )

    n = len(y)
    x = np.zeros(n)
    explanation = ""

    for i in reversed(range(n)):
        if abs(U[i, i]) < 1e-12:
            raise ValueError("Matrix is singular in back substitution.")

        sum_val = np.dot(U[i, i+1:], x[i+1:])
        x[i] = (y[i] - sum_val) / U[i, i]

        if help:
            explanation += (
                f"Solving x[{i}]:\n"
                f"sum = {sum_val}\n"
                f"x[{i}] = (y[{i}] - {sum_val}) / {U[i, i]} = {x[i]}\n\n"
            )

    if help:
        explanation += f"Complete solution vector x:\n{x}\n"

    return x, explanation, definition



def solve(A, b, help=False, what=False):

    definition = ""
    if what:
        definition += (
            "solve(A, b):\n"
            "This function solves a system of linear equations of the form Ax = b.\n"
            "It first converts A into an upper-triangular matrix (Gaussian elimination),\n"
            "and then solves it step-by-step (back substitution).\n\n"
            "Simple Example:\n"
            "A = [[2, 1], [4, 3]]\n"
            "b = [5, 11]\n"
            "The solution is x = [1, 3].\n"
        )

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    explanation = ""

    U, y, elim_expl, _ = gaussian_elimination(A.copy(), b.copy(), help=help, what=False)
    x, back_expl, _ = back_substitution(U, y, help=help, what=False)

    if help:
        explanation += "Step 1: Gaussian elimination\n"
        explanation += elim_expl
        explanation += "\nStep 2: Back substitution\n"
        explanation += back_expl

    return x, explanation, definition
