# ==============================
# Tutorial 4: Shape Manipulations
# ==============================
"""
This tutorial covers modifying matrix shapes: transposition, reshaping,
flattening, and stacking.
"""

from mllense.math.linalg import (
    transpose,
    reshape,
    flatten,
    vstack,
    hstack,
)

def run_tutorial():
    print("=== Linalg Tutorial: Shape Operations ===")

    A = [[1.0, 2.0, 3.0],
         [4.0, 5.0, 6.0]]
         
    print("Original 2x3 Matrix A:")
    for row in A: print("  ", row)

    # 1. Transpose
    # Swaps rows and columns (2x3 becomes 3x2)
    A_t = transpose(A)
    print("\n1. transpose(A):")
    for row in A_t: print("  ", row)

    # 2. Reshape
    # Changes dimensions while preserving the flattened order of elements
    # Let's reshape 2*3=6 into 3x2
    A_reshaped = reshape(A, 3, 2)
    print("\n2. reshape(A, 3, 2):")
    for row in A_reshaped: print("  ", row)

    # Let's reshape into 6x1
    A_reshaped_6_1 = reshape(A, 6, 1)
    print("\n   reshape(A, 6, 1):")
    for row in A_reshaped_6_1: print("  ", row)

    # 3. Flatten
    # Smashes the 2D matrix down into a 1-Dimensional list
    A_flat = flatten(A)
    print("\n3. flatten(A):")
    print("  ", A_flat)

    # 4. Vertical Stacking (vstack)
    # Stacks matrices vertically (row-wise)
    v_stacked = vstack(A, A)
    print("\n4. vstack(A, A):")
    for row in v_stacked: print("  ", row)

    # 5. Horizontal Stacking (hstack)
    # Stacks matrices horizontally (column-wise)
    h_stacked = hstack(A, A)
    print("\n5. hstack(A, A):")
    for row in h_stacked: print("  ", row)


    # --- Educational Expose ---
    print("\n=== EXPLAINABILITY SHOWCASE ===")
    from mllense.math.linalg import eye
    demo = eye(2, how_lense=True)
    print("- [WHAT LENSE]:", demo.what_lense.split('\n')[0])
    print("- [HOW LENSE]:", demo.how_lense.split('\n')[0], "...")

if __name__ == "__main__":
    print('\nRunning with Educational Engine v2.0\n')
run_tutorial()
