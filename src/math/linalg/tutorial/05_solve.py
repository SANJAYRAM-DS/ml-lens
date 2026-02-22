# ==============================
# Tutorial 5: Linear Systems (Solve)
# ==============================
"""
This tutorial covers solving linear systems of equations of the form `Ax = b`.
"""

from mllense.math.linalg import solve

def run_tutorial():
    print("=== Linalg Tutorial: Solving Linear Systems ===")

    # System of Equations:
    # 2x + 1y = 4
    # 5x + 3y = 7
    # 
    # Exact Solution: x = 5, y = -6
    
    A = [[2.0, 1.0], 
         [5.0, 3.0]]
    b = [4.0, 7.0]

    print("Coefficient Matrix A:")
    for row in A: print("  ", row)
    
    print("\nTarget Vector b:")
    print("  ", b)

    # Solve the system
    # Depending on the execution context, this will use Gaussian Elimination with 
    # Partial Pivoting or LU Decomposition underneath dynamically.
    x = solve(A, b)
    
    print("\nSolution Vector x (Ax = b):")
    print(f"  x = {x[0]:.2f}")
    print(f"  y = {x[1]:.2f}")
    
    # Verify:
    print("\nVerification (A @ x):")
    ax0 = A[0][0]*x[0] + A[0][1]*x[1]
    ax1 = A[1][0]*x[0] + A[1][1]*x[1]
    print(f"  Row 1: {ax0:.2f} (Expected {b[0]})")
    print(f"  Row 2: {ax1:.2f} (Expected {b[1]})")


    # --- Educational Expose ---
    print("\n=== EXPLAINABILITY SHOWCASE ===")
    from mllense.math.linalg import eye
    demo = eye(2, how_lense=True)
    print("- [WHAT LENSE]:", demo.what_lense.split('\n')[0])
    print("- [HOW LENSE]:", demo.how_lense.split('\n')[0], "...")

if __name__ == "__main__":
    print('\nRunning with Educational Engine v2.0\n')
run_tutorial()
