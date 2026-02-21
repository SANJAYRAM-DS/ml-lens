# ==============================
# Tutorial 7: Norms and Eigen Methods
# ==============================
"""
This tutorial covers computing Vector and Matrix norms, alongside
custom algorithms like finding the dominant eigenpair.
"""

from mllense.math.linalg import vector_norm, frobenius_norm, spectral_norm, dominant_eigen

def run_tutorial():
    print("=== Linalg Tutorial: Norms and Eigen Methods ===")

    # ── 1. Vector Norms ──
    v = [3.0, -4.0, 5.0]
    print(f"Vector v: {v}")
    
    # L2 Norm (Euclidean distance) - Default
    l2 = vector_norm(v, ord=2)
    print(f"  L2 Norm: {l2:.2f}")
    
    # L1 Norm (Manhattan distance / sum of absolutes)
    l1 = vector_norm(v, ord=1)
    print(f"  L1 Norm: {l1:.2f}")
    
    # L-Infinity Norm (Maximum absolute value)
    linf = vector_norm(v, ord=float("inf"))
    print(f"  L-Inf Norm: {linf:.2f}")


    # ── 2. Matrix Norms ──
    M = [[1.0, -2.0],
         [3.0, 4.0]]
    print("\nMatrix M:")
    for row in M: print("  ", row)
    
    # Frobenius norm: sqrt of sum of squares of elements
    f_norm = frobenius_norm(M)
    print(f"  Frobenius Norm: {f_norm:.2f}")
    
    # Spectral norm: largest singular value (2-norm)
    # Computes SVD internally
    s_norm = spectral_norm(M)
    print(f"  Spectral (L2) Norm: {s_norm:.2f}")


    # ── 3. Dominant Eigenpair ──
    # Computes the largest eigenvalue and its corresponding eigenvector
    # using the native Power Iteration algorithm. No numpy required!
    S = [[2.0, 1.0], 
         [1.0, 2.0]]
    
    print("\nSymmetric Matrix S:")
    for row in S: print("  ", row)
    
    eigenvalue, eigenvector = dominant_eigen(S)
    print(f"  Dominant Eigenvalue: {eigenvalue:.5f}")
    print(f"  Dominant Eigenvector: {[round(x, 5) for x in eigenvector]}")

if __name__ == "__main__":
    run_tutorial()
