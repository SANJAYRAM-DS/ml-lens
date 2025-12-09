import random

def zeros(shape, help=False, what=False):
    rows, cols = shape
    result = [[0 for _ in range(cols)] for _ in range(rows)]

    explanation = ""
    if help:
        explanation += f"Step-by-step: Creating a {rows}x{cols} zero matrix.\n"
        explanation += "Each element is set to 0.\n"

    definition = ""
    if what:
        definition += (
            "The `zeros` function creates a matrix filled entirely with 0s.\n"
            f"Example: zeros(({rows}, {cols})) → {result}\n"
        )

    return result, explanation, definition


def ones(shape, help=False, what=False):
    rows, cols = shape
    result = [[1 for _ in range(cols)] for _ in range(rows)]

    explanation = ""
    if help:
        explanation += f"Step-by-step: Creating a {rows}x{cols} matrix of ones.\n"
        explanation += "Each element is set to 1.\n"

    definition = ""
    if what:
        definition += (
            "The `ones` function creates a matrix in which every value is 1.\n"
            f"Example: ones(({rows}, {cols})) → {result}\n"
        )

    return result, explanation, definition


def eye(n, help=False, what=False):
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    explanation = ""
    if help:
        explanation += f"Step-by-step: Creating a {n}x{n} identity matrix.\n"
        explanation += "1s on the diagonal, 0s elsewhere.\n"

    definition = ""
    if what:
        definition += (
            "The `eye` function creates an identity matrix.\n"
            "This is a square matrix with 1s on the diagonal.\n"
            f"Example: eye({n}) → {result}\n"
        )

    return result, explanation, definition


def rand(shape, seed=None, help=False, what=False):
    if seed is not None:
        random.seed(seed)

    rows, cols = shape
    result = [[random.random() for _ in range(cols)] for _ in range(rows)]

    explanation = ""
    if help:
        explanation += f"Step-by-step: Creating a {rows}x{cols} random matrix.\n"
        explanation += "Each value is a random float between 0 and 1.\n"

    definition = ""
    if what:
        definition += (
            "The `rand` function creates a matrix filled with random numbers.\n"
            "Values are between 0 and 1.\n"
            f"Example: rand(({rows}, {cols})) → {result}\n"
        )

    return result, explanation, definition
