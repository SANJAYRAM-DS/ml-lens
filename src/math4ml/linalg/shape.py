def flatten(A, help=False):
    result = [x for row in A for x in row]
    
    explanation = ""
    if help:
        explanation += "Flatten:\n"
        explanation += "Step 1: Take every row and extract elements one by one.\n"
        explanation += f"Original matrix: {A}\n"
        explanation += f"Flattened output: {result}\n"
    
    return result, explanation


def reshape(A, new_shape, help=False):
    rows, cols = new_shape
    flat = [x for row in A for x in row]

    if len(flat) != rows * cols:
        raise ValueError("Cannot reshape: total elements mismatch.")

    result = [flat[i*cols:(i+1)*cols] for i in range(rows)]

    explanation = ""
    if help:
        explanation += "Reshape:\n"
        explanation += "Step 1: Flatten the matrix.\n"
        explanation += f"Flattened: {flat}\n"
        explanation += f"Step 2: Split into {rows} rows each of size {cols}.\n"
        explanation += f"Reshaped matrix:\n{result}\n"
    
    return result, explanation


def stack(A, B, axis=0, help=False):
    if axis == 0:
        result = A + B
    elif axis == 1:
        if len(A) != len(B):
            raise ValueError("Matrices must have same number of rows for axis=1")
        result = [rowA + rowB for rowA, rowB in zip(A, B)]
    else:
        raise ValueError("axis must be 0 or 1")

    explanation = ""
    if help:
        explanation += "Stack:\n"
        if axis == 0:
            explanation += "Axis 0 → Put matrix B below matrix A (vertical stacking).\n"
        else:
            explanation += "Axis 1 → Put matrix B beside matrix A (horizontal stacking).\n"
        explanation += f"Matrix A: {A}\n"
        explanation += f"Matrix B: {B}\n"
        explanation += f"Result:\n{result}\n"

    return result, explanation


def concat(arrays, axis=0, help=False):
    if axis == 0:
        result = []
        for arr in arrays:
            result.extend(arr)
    elif axis == 1:
        row_count = len(arrays[0])
        for arr in arrays:
            if len(arr) != row_count:
                raise ValueError("All matrices must have same number of rows for axis=1")

        result = [
            sum((arr[i] for arr in arrays), [])
            for i in range(row_count)
        ]
    else:
        raise ValueError("axis must be 0 or 1")

    explanation = ""
    if help:
        explanation += "Concat:\n"
        if axis == 0:
            explanation += "Axis 0 → Append rows of each matrix vertically.\n"
        else:
            explanation += "Axis 1 → Append columns of each matrix horizontally.\n"

        for idx, arr in enumerate(arrays):
            explanation += f"Matrix {idx}: {arr}\n"

        explanation += f"Final concatenated result:\n{result}\n"

    return result, explanation
