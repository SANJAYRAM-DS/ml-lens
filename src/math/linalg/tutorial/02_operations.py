# ==============================
# Tutorial 2: Element-wise Operations
# ==============================
"""
This tutorial covers the element-wise operations available in the `ops` API.
All operations are performed element-by-element across matrices or vectors of the SAME shape,
or between a matrix/vector and a scalar value.
"""

from mllense.math.linalg import (
    add,
    subtract,
    multiply,
    divide,
    scalar_add,
    scalar_multiply,
)

def run_tutorial():
    print("=== Linalg Tutorial: Element-wise Operations ===")

    A = [[1.0, 2.0], 
         [3.0, 4.0]]
         
    B = [[5.0, 6.0], 
         [7.0, 8.0]]

    print("Matrix A:")
    for row in A: print("  ", row)
    
    print("Matrix B:")
    for row in B: print("  ", row)

    # 1. Addition
    res_add = add(A, B)
    print("\n1. add(A, B):")
    for row in res_add: print("  ", row)

    # 2. Subtraction
    res_sub = subtract(B, A)
    print("\n2. subtract(B, A):")
    for row in res_sub: print("  ", row)

    # 3. Element-wise Multiplication (Hadamard product)
    # Note: This is NOT matrix multiplication! 
    res_mul = multiply(A, B)
    print("\n3. multiply(A, B)  [Hadamard]:")
    for row in res_mul: print("  ", row)

    # 4. Element-wise Division
    res_div = divide(B, A)
    print("\n4. divide(B, A):")
    for row in res_div: print("  ", row)

    # 5. Scalar Operations
    # Add a number to every element
    res_sadd = scalar_add(A, 10.0)
    print("\n5. scalar_add(A, 10.0):")
    for row in res_sadd: print("  ", row)

    # Multiply every element by a number
    res_smul = scalar_multiply(A, 0.5)
    print("\n6. scalar_multiply(A, 0.5):")
    for row in res_smul: print("  ", row)

if __name__ == "__main__":
    run_tutorial()
