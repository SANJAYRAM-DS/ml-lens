import numpy as np

def transpose(A, help=False, what=False):
    result = [list(row) for row in zip(*A)]
    explanation = ""

    if what:
        explanation += "WHAT: The transpose flips a matrix over its diagonal.\n"
        explanation += "Rows become columns.\n"
        explanation += f"Example: transpose({A}) = {result}\n"
        return result, explanation

    if help:
        explanation += "WHAT IT DOES:\n"
        explanation += "The transpose operation flips the matrix over its diagonal.\n"
        explanation += "Rows become columns and columns become rows.\n\n"

        explanation += "EXAMPLE:\n"
        explanation += f"Input A = {A}\n"
        explanation += f"Output Transpose(A) = {result}\n\n"

        explanation += "STEP-BY-STEP:\n"
        explanation += "Step 1: Read each row of A.\n"
        explanation += "Step 2: Turn the first column of A into the first row.\n"
        explanation += "Step 3: Continue for all columns.\n"

    return result, explanation


def dot(a, b, help=False, what=False):
    a = np.array(a).flatten()
    b = np.array(b).flatten()

    if a.shape[0] != b.shape[0]:
        raise ValueError("Vectors must have the same length")

    result = np.sum(a * b)
    explanation = ""

    if what:
        explanation += "WHAT: Dot product multiplies matching elements and adds them.\n"
        explanation += f"Example: dot({a.tolist()}, {b.tolist()}) = {result}\n"
        return result, explanation

    if help:
        explanation += "WHAT IT DOES:\n"
        explanation += "The dot product multiplies matching elements and adds them up.\n\n"

        explanation += "EXAMPLE:\n"
        explanation += f"a = {a.tolist()}, b = {b.tolist()}\n"
        explanation += f"dot(a, b) = {result}\n\n"

        explanation += "STEP-BY-STEP:\n"
        for i in range(len(a)):
            explanation += f"  - a[{i}] * b[{i}] = {a[i]} * {b[i]} = {a[i]*b[i]}\n"
        explanation += f"Sum of all products = {result}\n"

    return result, explanation


def outer(a, b, help=False, what=False):
    result = np.outer(a, b).tolist()
    explanation = ""

    if what:
        explanation += "WHAT: Outer product multiplies each element of a with every element of b.\n"
        explanation += f"Example: outer({a}, {b}) = {result}\n"
        return result, explanation

    if help:
        explanation += "WHAT IT DOES:\n"
        explanation += "Creates a matrix by multiplying each element of vector a with each element of vector b.\n\n"

        explanation += "EXAMPLE:\n"
        explanation += f"a = {a}, b = {b}\n"
        explanation += f"outer(a, b) = {result}\n\n"

        explanation += "STEP-BY-STEP:\n"
        for i in range(len(a)):
            for j in range(len(b)):
                explanation += f"  - a[{i}] * b[{j}] = {a[i]} * {b[j]} = {result[i][j]}\n"

    return result, explanation


def matmul(A, B, help=False, what=False):
    A_np = np.array(A)
    B_np = np.array(B)

    if A_np.shape[1] != B_np.shape[0]:
        raise ValueError("Shapes not aligned for multiplication")

    result = (A_np @ B_np).tolist()
    explanation = ""

    if what:
        explanation += "WHAT: Matrix multiplication multiplies rows of A with columns of B.\n"
        explanation += f"Example: matmul({A}, {B}) = {result}\n"
        return result, explanation

    if help:
        explanation += "WHAT IT DOES:\n"
        explanation += "Performs matrix multiplication: row of A × column of B.\n\n"

        explanation += "EXAMPLE:\n"
        explanation += f"A = {A}\n"
        explanation += f"B = {B}\n"
        explanation += f"A @ B = {result}\n\n"

        explanation += "STEP-BY-STEP:\n"
        m, n = len(A), len(A[0])
        p = len(B[0])

        for i in range(m):
            for j in range(p):
                explanation += f"Entry ({i},{j}): "
                explanation += " + ".join([f"{A[i][k]}*{B[k][j]}" for k in range(n)])
                explanation += f" = {result[i][j]}\n"

    return result, explanation
