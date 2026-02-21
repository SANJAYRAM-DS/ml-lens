# ==============================
# Tutorial 3: Matrix Multiplication
# ==============================
"""
This tutorial demonstrates standard mathematical Matrix Multiplication using the `matmul` API.
The `matmul` function is the powerhouse of the linalg core and scales internally by
choosing the most optimized algorithm (block, strassen, or naive) based on matrix dimensions.
"""

from mllense.math.linalg import matmul

def run_tutorial():
    print("=== Linalg Tutorial: Matrix Multiplication ===")

    # ── 1. Inner Product / Dot Product (1D x 1D) ──
    v1 = [1.0, 2.0, 3.0]
    v2 = [4.0, 5.0, 6.0]
    
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    dot = matmul(v1, v2)
    print("\n1. 1D x 1D (Dot Product):")
    print(f"  {v1} @ {v2} = {dot}")

    # ── 2. Matrix x Vector (2D x 1D) ──
    # Matrix A is 2x3, Vector x is 3
    A = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
    x = [1.0, 0.0, -1.0]
    
    # Result should be a 1D vector of length 2
    res_vec = matmul(A, x)
    print("\n2. 2D x 1D (Matrix-Vector Multiplication):")
    print(f"  {A} \n     @ \n  {x} \n  = {res_vec}")

    # ── 3. Matrix x Matrix (2D x 2D) ──
    # Matrix M1 is 2x3, Matrix M2 is 3x2. Result is 2x2.
    M1 = [[1.0, 2.0, 3.0], 
          [4.0, 5.0, 6.0]]
    M2 = [[7.0, 8.0],
          [9.0, 10.0],
          [11.0, 12.0]]
          
    res_mat = matmul(M1, M2)
    print("\n3. 2D x 2D (Matrix-Matrix Multiplication):")
    print("  Resulting Matrix:")
    for row in res_mat:
        print("    ", row)

    # ── 4. Explicit Backends ──
    # You can forcibly execute via pure Python or hand off to NumPy manually.
    import numpy as np
    a_np = np.array([[1.0, 2.0], [3.0, 4.0]])
    b_np = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    # Passing numpy arrays returns a numpy array!
    out_np = matmul(a_np, b_np)
    print("\n4. Automatic NumPy handling:")
    print("  Returned type:", type(out_np))
    print(out_np)

if __name__ == "__main__":
    run_tutorial()
