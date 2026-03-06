import numpy as np

def trace(A, help=False, what=False):
    A_np = np.array(A)
    result = np.trace(A_np)

    explanation = ""
    if help:
        explanation += "Step-by-step: To compute the trace, add the diagonal elements.\n"
        for i in range(len(A)):
            explanation += f"A[{i}][{i}] = {A[i][i]}\n"
        explanation += f"Trace = {result}\n"

    definition = ""
    if what:
        definition += (
            "The `trace` function returns the sum of the diagonal elements of a square matrix.\n"
            f"Example: For matrix {A}, trace = {result}.\n"
        )

    return result, explanation, definition


def det(A, help=False, what=False):
    A_np = np.array(A)
    result = round(np.linalg.det(A_np), 6)

    explanation = ""
    if help:
        explanation += "Step-by-step: Computing determinant.\n"
        explanation += "Matrix:\n"
        for row in A:
            explanation += f"{row}\n"
        explanation += f"Determinant = {result}\n"

    definition = ""
    if what:
        definition += (
            "The `det` function returns the determinant of a square matrix.\n"
            "The determinant helps check if the matrix is invertible.\n"
            f"Example: det({A}) = {result}\n"
        )

    return result, explanation, definition


def rank(A, help=False, what=False):
    A_np = np.array(A)
    result = np.linalg.matrix_rank(A_np)

    explanation = ""
    if help:
        explanation += "Step-by-step: Rank shows the number of independent rows/columns.\n"
        explanation += "Matrix:\n"
        for row in A:
            explanation += f"{row}\n"
        explanation += f"Rank = {result}\n"

    definition = ""
    if what:
        definition += (
            "The `rank` function tells you how many rows or columns in a matrix are independent.\n"
            "A full-rank matrix means no row/column is a combination of others.\n"
            f"Example: rank({A}) = {result}\n"
        )

    return result, explanation, definition


def inverse(A, help=False, what=False):
    A_np = np.array(A)
    result = np.linalg.inv(A_np).tolist()

    explanation = ""
    if help:
        explanation += "Step-by-step: Computing inverse of the matrix.\n"
        explanation += "Original matrix:\n"
        for row in A:
            explanation += f"{row}\n"
        explanation += "Inverse matrix:\n"
        for row in result:
            explanation += f"{row}\n"

    definition = ""
    if what:
        definition += (
            "The `inverse` function returns the inverse of a square matrix.\n"
            "A matrix is invertible only if its determinant is not zero.\n"
            f"Example: inverse({A}) = {result}\n"
        )

    return result, explanation, definition
