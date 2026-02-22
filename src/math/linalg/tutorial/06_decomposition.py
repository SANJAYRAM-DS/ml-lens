# ==============================
# Tutorial 6: Matrix Decomposition
# ==============================
"""
This tutorial covers matrix properties, inversion, and decomposition algorithms.
"""

from mllense.math.linalg import (
    det,
    inv,
    matrix_trace,
    qr,
    svd,
    eig
)

def run_tutorial():
    print("=== Linalg Tutorial: Decomposition and Properties ===")

    A = [[1.0, 2.0], 
         [3.0, 4.0]]
    
    print("Square Matrix A:")
    for row in A: print("  ", row)

    # 1. Determinant
    # Computed via LU decomposition behind the scenes.
    determinant = det(A)
    print(f"\n1. det(A): {determinant:.2f}")

    # 2. Trace
    # Sum of the diagonal elements
    trace_val = matrix_trace(A)
    print(f"\n2. matrix_trace(A): {trace_val:.2f}")

    # 3. Inverse
    # Finds A^-1 using Gauss-Jordan elimination.
    A_inv = inv(A)
    print("\n3. inv(A):")
    for row in A_inv: print(f"  [{row[0]:>6.2f}, {row[1]:>6.2f}]")

    # 4. QR Decomposition
    # A = Q * R, where Q is orthogonal, R is upper triangular.
    M = [[12.0, -51.0, 4.0],
         [6.0, 167.0, -68.0],
         [-4.0, 24.0, -41.0]]
    print("\nMatrix M (for QR):")
    for row in M: print(f"  {row}")
    
    Q, R = qr(M)
    print("\n4. qr(M):")
    print("  Q factor:")
    for row in Q: print(f"    {[round(x, 2) for x in row]}")
    print("  R factor:")
    for row in R: print(f"    {[round(x, 2) for x in row]}")

    # 5. SVD & Eigendecomposition (Wraps Numpy inherently)
    U, S, Vt = svd(A)
    print("\n5. svd(A):")
    print("  Singular Values (S):", [round(s, 2) for s in S])
    
    try:
        vals, vecs = eig(A)
        print("\n6. eig(A):")
        print("  Eigenvalues:", [round(x, 2) for x in vals])
    except Exception as e:
        print(f"Error calling eig: {e}")


    # --- Educational Expose ---
    print("\n=== EXPLAINABILITY SHOWCASE ===")
    from mllense.math.linalg import eye
    demo = eye(2, how_lense=True)
    print("- [WHAT LENSE]:", demo.what_lense.split('\n')[0])
    print("- [HOW LENSE]:", demo.how_lense.split('\n')[0], "...")

if __name__ == "__main__":
    print('\nRunning with Educational Engine v2.0\n')
run_tutorial()
