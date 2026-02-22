# ==============================
# Tutorial 8: Matrix Diagnostics
# ==============================
"""
This tutorial showcases the robust diagnostic capabilities for evaluating
the stability, rank, and condition number of mathematical matrices before
plugging them into solving architectures.
"""

from mllense.math.linalg import (
    condition_number, 
    matrix_rank, 
    stability_report, 
    full_diagnostic_report
)

def run_tutorial():
    print("=== Linalg Tutorial: Diagnostics & Stability ===")

    # A stable, full-rank matrix
    A_stable = [[2.0, 1.0], 
                [1.0, 2.0]]
    
    # A perfectly singular matrix (rank 1 instead of 2)
    A_singular = [[1.0, 2.0], 
                  [2.0, 4.0]]
                  
    print("Stable Matrix:")
    for row in A_stable: print("  ", row)

    # 1. Condition Number
    # Defines how sensitive the function is to changes or errors in the input.
    # Lower is better (1.0 is ideal).
    cond_stable = condition_number(A_stable)
    cond_singular = condition_number(A_singular)
    
    print(f"\n1. Condition Number (Stable): {cond_stable:.2f}")
    print(f"   Condition Number (Singular): {cond_singular}") # Will be math.inf

    # 2. Matrix Rank
    # Number of linearly independent row/column vectors.
    r_stable = matrix_rank(A_stable)
    r_singular = matrix_rank(A_singular)
    
    print(f"\n2. Matrix Rank (Stable): {r_stable} (out of 2)")
    print(f"   Matrix Rank (Singular): {r_singular} (out of 2)")

    # 3. Stability Report (Dictionary)
    # Packages these metrics neatly.
    rep = stability_report(A_singular)
    print("\n3. Stability Report for Singular Matrix:")
    for k, v in rep.items():
        print(f"  {k}: {v}")

    # 4. Generate Comprehensive Diagnostic Log
    print("\n4. Comprehensive Diagnostic Log for Stable Matrix:")
    full_log = full_diagnostic_report(A_stable)
    print(full_log)


    # --- Educational Expose ---
    print("\n=== EXPLAINABILITY SHOWCASE ===")
    from mllense.math.linalg import eye
    demo = eye(2, how_lense=True)
    print("- [WHAT LENSE]:", demo.what_lense.split('\n')[0])
    print("- [HOW LENSE]:", demo.how_lense.split('\n')[0], "...")

if __name__ == "__main__":
    print('\nRunning with Educational Engine v2.0\n')
run_tutorial()
