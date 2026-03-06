def _check_same_shape(A, B):
    if len(A) != len(B):
        raise ValueError("Matrices must have the same number of rows.")
    for row_a, row_b in zip(A, B):
        if len(row_a) != len(row_b):
            raise ValueError("Matrices must have the same number of columns.")


def add(A, B, help=False, what=False):
    _check_same_shape(A, B)
    result = [[a+b for a,b in zip(rowA,rowB)] for rowA,rowB in zip(A,B)]
    explanation = ""

    if what:
        explanation += "WHAT: Matrix addition adds each element of A with the matching element of B.\n"
        explanation += f"Example: add({A}, {B}) = {result}\n"
        return result, explanation

    if help:
        explanation += "Step 1: Add each element from A with the corresponding element in B:\n"
        for i in range(len(A)):
            for j in range(len(A[0])):
                explanation += f"A[{i}][{j}] + B[{i}][{j}] = {A[i][j]} + {B[i][j]} = {result[i][j]}\n"
        explanation += f"Step 2: Result = {result}"

    return result, explanation


def subtract(A, B, help=False, what=False):
    _check_same_shape(A, B)
    result = [[a-b for a,b in zip(rowA,rowB)] for rowA,rowB in zip(A,B)]
    explanation = ""

    if what:
        explanation += "WHAT: Matrix subtraction subtracts each element of B from A.\n"
        explanation += f"Example: subtract({A}, {B}) = {result}\n"
        return result, explanation

    if help:
        explanation += "Step 1: Subtract each element in B from A:\n"
        for i in range(len(A)):
            for j in range(len(A[0])):
                explanation += f"{A[i][j]} - {B[i][j]} = {result[i][j]}\n"
        explanation += f"Step 2: Result = {result}"

    return result, explanation


def multiply(A, B, help=False, what=False):
    _check_same_shape(A, B)
    result = [[a*b for a,b in zip(rowA,rowB)] for rowA,rowB in zip(A,B)]
    explanation = ""

    if what:
        explanation += "WHAT: Element-wise multiplication multiplies matching elements of A and B.\n"
        explanation += f"Example: multiply({A}, {B}) = {result}\n"
        return result, explanation

    if help:
        explanation += "Step 1: Multiply each element of A by the corresponding element in B:\n"
        for i in range(len(A)):
            for j in range(len(A[0])):
                explanation += f"{A[i][j]} * {B[i][j]} = {result[i][j]}\n"
        explanation += f"Step 2: Result = {result}"

    return result, explanation


def divide(A, B, help=False, what=False):
    _check_same_shape(A, B)
    result = [[a/b for a,b in zip(rowA,rowB)] for rowA,rowB in zip(A,B)]
    explanation = ""

    if what:
        explanation += "WHAT: Element-wise division divides A’s elements by matching elements in B.\n"
        explanation += f"Example: divide({A}, {B}) = {result}\n"
        return result, explanation

    if help:
        explanation += "Step 1: Divide each element in A by the corresponding element in B:\n"
        for i in range(len(A)):
            for j in range(len(A[0])):
                explanation += f"{A[i][j]} / {B[i][j]} = {result[i][j]}\n"
        explanation += f"Step 2: Result = {result}"

    return result, explanation


def scalar_multiply(A, s, help=False, what=False):
    result = [[a*s for a in row] for row in A]
    explanation = ""

    if what:
        explanation += "WHAT: Scalar multiplication multiplies every element of A by a number.\n"
        explanation += f"Example: scalar_multiply({A}, {s}) = {result}\n"
        return result, explanation

    if help:
        explanation += f"Step 1: Multiply each element in A by scalar {s}:\n"
        for i, row in enumerate(A):
            for j, val in enumerate(row):
                explanation += f"{val} * {s} = {result[i][j]}\n"
        explanation += f"Step 2: Result = {result}"

    return result, explanation
